"""
Federated Learning with Byzantine-Robust Defense (v7)

This module implements a robust federated learning system with:
- Cluster-agnostic Byzantine detection using Subjective Logic
- Soft-Dropout aggregation for attack resilience
- Temporal centroid momentum for stable clustering
- ASR-based reputation penalties for known attacks
- Fairness-aware reward distribution

Author: Project Team
Version: 7
"""

import os
import sys
import shutil
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.multiprocessing as mp
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


# ==============================================================================
# CONFIGURATION
# ==============================================================================

class Config:
    """Global configuration for the federated learning experiment."""

    # Hardware & System
    MAX_WORKERS = 2
    OUTPUT_DIR = "cifar10_v10_results"
    DEVICE_LIST = (
        ['cuda:{}'.format(i) for i in range(torch.cuda.device_count())]
        if torch.cuda.is_available()
        else ['cpu']
    )

    # Federated Learning Parameters
    NUM_CLIENTS = 10
    ROUNDS = 30
    SEED = 1234
    MALICIOUS_COUNTS = [0, 2, 4, 6]
    MAL_TYPE = 'label_flip'  # 'label_flip' or 'random_grad'

    # Local Training Parameters
    LOCAL_EPOCHS = 1
    BATCH_SIZE = 64
    LR = 0.01
    MOMENTUM = 0.9
    WEIGHT_DECAY = 5e-4

    # Subjective Logic & Reputation Parameters
    EPS = 1e-10
    REPUTATION_BETA = 0.05
    NU = 0.5
    DELTA = 0.85
    W_FIXED = 0.1
    RHO_MIN = 0.9
    PF_FRACTION = 0.05
    LAMBDA_PENALTY = 2.0
    TAU_PENALTY = 0.2
    REWARD_Q = 1.0

    # Z-Score Calibration (Safety Floor)
    # Prevents benign NonIID outliers from permanent exclusion
    ZSCORE_FLOOR = 0.15

    # Soft-Dropout Aggregation
    # Primary channel: top 50% by reputation (weight = alpha)
    # Residual channel: all clients via inverse-uncertainty (weight = 1 - alpha)
    SOFT_DROPOUT_ALPHA = 0.7

    # Clustering State (Updated per round)
    PREV_CENTROIDS = None 


# ==============================================================================
# MODEL DEFINITION
# ==============================================================================

class SimpleCNN(nn.Module):
    """Simple CNN for CIFAR-10 classification.

    Architecture:
    - 3 convolutional blocks with batch normalization and max pooling
    - 2 fully connected layers with dropout-like regularization
    """

    def __init__(self, num_classes: int = 10):
        """Initialize the CNN model.

        Args:
            num_classes: Number of output classes (default: 10 for CIFAR-10)
        """
        super().__init__()
        self.net = nn.Sequential(
            # Block 1: 32x32 -> 16x16
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 2: 16x16 -> 8x8
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 3: 8x8 -> 4x4
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Fully connected layers
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, 3, 32, 32)

        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        return self.net(x)



# ==============================================================================
# CLUSTERING AND BYZANTINE DETECTION
# ==============================================================================

def compute_cluster_opinions(
    updates: list,
    global_grad_ref: dict,
    n_clusters: int = 3, 
    outlier_threshold: float = 1.5,
    rnd: int = 0,
) -> list:
    """
    NOVELTY: Salient Feature Alignment (SFA).
    
    1. Identify 'Salient' weights (Top 20% magnitude) from Server Reference.
    2. Compute Cosine Similarity ONLY on these salient weights.
    3. This ignores Non-IID noise in unimportant weights and prevents 
       scaling attacks (via Cosine Normalization).
    """
    n_clients = len(updates)
    opinions = [None] * n_clients
    
    # Target the Final Linear Layer
    target_layer = 'net.15.weight'
    
    # 1. Extract and Flatten Server Reference
    server_vec = global_grad_ref[target_layer].flatten()
    
    # 2. Compute Saliency Mask (Dynamic Feature Selection)
    # We only care about the top 20% of weights that the server deems 'critical'
    # This filters out noise where Non-IID clients naturally disagree.
    threshold = np.percentile(np.abs(server_vec), 80)
    mask = np.abs(server_vec) > threshold
    
    # Safety: If mask is empty (rare), use all weights
    if np.sum(mask) == 0:
        mask[:] = True

    # Filter Server Vector
    server_salient = server_vec[mask]
    server_norm = np.linalg.norm(server_salient) + 1e-10

    scores = []
    for i, upd in enumerate(updates):
        client_vec = upd[target_layer].flatten()
        
        # Filter Client Vector
        client_salient = client_vec[mask]
        
        # 3. Compute Masked Cosine Similarity
        # Scale-invariant: Prevents attackers from using huge magnitudes to dominate
        client_norm = np.linalg.norm(client_salient) + 1e-10
        cosine_sim = np.dot(client_salient, server_salient) / (client_norm * server_norm)
        
        scores.append(cosine_sim)

    scores = np.array(scores)
    
    # 4. Map to Subjective Logic (Non-Linear Boosting)
    # We use a power function to heavily reward high alignment and punish mediocrity
    for i, score in enumerate(scores):
        if score <= 0:
            # OPPOSITION: Veto
            opinions[i] = {'b': 0.0, 'd': 1.0, 'u': 0.0}
        else:
            # ALIGNMENT:
            # Map Cosine [0, 1] -> Belief [0, 0.9]
            # Squaring the score (score^2) penalizes weak alignment (0.5 -> 0.25)
            # helping to distinguish "Mediocre" from "Excellent"
            b = (score ** 2) * 0.9
            u = 0.1
            d = max(0.0, 1.0 - b - u)
            
            opinions[i] = {'b': b, 'd': d, 'u': u}

    return opinions
# ==============================================================================
# SUBJECTIVE LOGIC AND AGGREGATION UTILITIES
# ==============================================================================

def fuse_opinions(op1: dict, op2: dict) -> dict:
    """Fuse two Subjective Logic opinions.

    Args:
        op1: Opinion 1 with keys 'b' (belief), 'd' (disbelief), 'u' (uncertainty)
        op2: Opinion 2 with same structure

    Returns:
        Fused opinion using Subjective Logic fusion rule
    """
    b1, d1, u1 = op1['b'], op1['d'], op1['u']
    b2, d2, u2 = op2['b'], op2['d'], op2['u']

    kappa = u1 + u2 - u1 * u2
    if kappa < 1e-9:
        kappa = 1e-9

    return {
        'b': (b1 * u2 + b2 * u1) / kappa,
        'd': (d1 * u2 + d2 * u1) / kappa,
        'u': (u1 * u2) / kappa,
    }


def aggregate_models_weighted(
    global_model: SimpleCNN,
    updates: list,
    weights: np.ndarray,
) -> None:
    """Aggregate client models using reputation-based weighting.

    Args:
        global_model: Global model to update
        updates: List of weight update tensors from clients
        weights: Aggregation weights (normalized to sum to 1)
    """
    global_dict = global_model.state_dict()
    new_state = {}

    # Compute weighted aggregation
    for k in updates[0].keys():
        agg_tensor = torch.zeros_like(updates[0][k])
        for i, update in enumerate(updates):
            agg_tensor += weights[i] * update[k]
        new_state[k] = global_dict[k] + agg_tensor.to(global_dict[k].device)

    global_model.load_state_dict(new_state)


# ==============================================================================
# CLIENT TRAINING WORKER
# ==============================================================================

def client_worker(args: tuple) -> tuple:
    """Independent worker function for local client training.

    Performs local training on client data, handles malicious attacks,
    and computes model update vectors.

    Args:
        args: Tuple containing:
            - client_idx (int): Client identifier
            - global_weights (dict): Global model weights to start from
            - indices (list): Local training data indices
            - malicious (bool): Whether client is malicious
            - mal_type (str): Type of attack ('label_flip' or 'random_grad')
            - device_name (str): Device to train on
            - worker_seed (int): Random seed for reproducibility
            - rnd (int): Current round number

    Returns:
        Tuple of (update_dict, loss_pre, loss_post) where:
            - update_dict: Weight update as numpy arrays
            - loss_pre: Validation loss before training
            - loss_post: Validation loss after training
    """
    (
        client_idx,
        global_weights,
        indices,
        malicious,
        mal_type,
        device_name,
        worker_seed,
        rnd,
    ) = args

    # Setup Environment
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)

    device = (
        torch.device(device_name)
        if torch.cuda.is_available() and "cuda" in device_name
        else torch.device("cpu")
    )

    # Data Transforms
    transform_local = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    transform_eval = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    # Load Datasets
    full_dataset_train = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=False, transform=transform_local
    )
    full_dataset_eval = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=False, transform=transform_eval
    )

    # Split indices for local robustness check
    split_point = max(int(len(indices) * 0.2), 50)
    val_indices = indices[:split_point]
    train_indices = indices[split_point:]

    loader_val = DataLoader(
        Subset(full_dataset_eval, val_indices),
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
    )

    # Handle Malicious Label Flip Attack
    if malicious and mal_type == 'label_flip':
        flip_from, flip_to = 0, 9
        flipped_data = []
        subset = Subset(full_dataset_train, train_indices)
        for i in range(len(subset)):
            x, y = subset[i]
            if int(y) == flip_from:
                y = flip_to
            flipped_data.append((x, y))
        loader_train = DataLoader(
            flipped_data, batch_size=Config.BATCH_SIZE, shuffle=True
        )
    else:
        loader_train = DataLoader(
            Subset(full_dataset_train, train_indices),
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
        )

    # Initialize Model
    local_model = SimpleCNN().to(device)
    local_model.load_state_dict(global_weights)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        local_model.parameters(),
        lr=Config.LR,
        momentum=Config.MOMENTUM,
        weight_decay=Config.WEIGHT_DECAY,
    )

    # Helper: Evaluate Validation Loss
    def get_loss(loader):
        local_model.eval()
        total_loss = 0.0
        count = 0
        with torch.no_grad():
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                out = local_model(xb)
                total_loss += criterion(out, yb).item() * xb.size(0)
                count += xb.size(0)
        return total_loss / max(1, count)

    # Pre-Training Validation
    loss_pre = get_loss(loader_val)

    # Local Training Loop
    local_model.train()
    for _ in range(Config.LOCAL_EPOCHS):
        for xb, yb in loader_train:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = local_model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()

    # Post-Training Validation
    loss_post = get_loss(loader_val)

    # Compute Update Vector
    update = {}
    local_state = local_model.state_dict()
    for k, v_global in global_weights.items():
        v_local = local_state[k].cpu()
        update[k] = (v_local - v_global).numpy().astype(np.float32)

    # Handle Random Gradient Attack
    if malicious and mal_type == 'random_grad':
        for k in update:
            update[k] = np.random.randn(*update[k].shape).astype(np.float32) * 0.05

    return update, loss_pre, loss_post


# ==============================================================================
# EXPERIMENT CONTROLLER
# ==============================================================================

def run_experiment(
    partition: list,
    mal_count: int,
    test_data,
    loader_asr,
    loader_server_ref,
    mal_type: str = 'label_flip',
    tag: str = 'iid',
) -> dict:
    server_device = torch.device(Config.DEVICE_LIST[0])
    global_model = SimpleCNN().to(server_device)

    n_clients = len(partition)
    reputation = np.ones(n_clients) / n_clients 

    # Logging
    logs = {
        'acc_log': [], 'asr_log': [], 'fairness_log': [], 'rewards_log': [],
        'reputations_log': [], 'avg_rep_mal_log': [], 'avg_rep_ben_log': [],
        'convergence_round_log': None,
    }

    # Setup Malicious Flags
    malicious_flags = [False] * n_clients
    mal_idxs = np.random.choice(n_clients, mal_count, replace=False)
    for i in mal_idxs: malicious_flags[i] = True

    print(f"Starting {tag} (Malicious: {mal_count})")

    with mp.Pool(processes=Config.MAX_WORKERS) as pool:
        for rnd in range(Config.ROUNDS):

            # --- Step A: Compute Server Reference Gradient (The Gold Standard) ---
            # We train the global model on the small server dataset to get the "True Direction"
            global_state_cpu = {k: v.cpu() for k, v in global_model.state_dict().items()}
            state_backup = {k: v.cpu().clone() for k, v in global_model.state_dict().items()}

            global_model.train()
            server_opt = optim.SGD(global_model.parameters(), lr=Config.LR, momentum=Config.MOMENTUM)

            for xb, yb in loader_server_ref:
                xb, yb = xb.to(server_device), yb.to(server_device)
                server_opt.zero_grad()
                out = global_model(xb)
                loss = nn.CrossEntropyLoss()(out, yb)
                loss.backward()
                server_opt.step()

            state_after_server = global_model.state_dict()
            global_grad_ref = {}
            for k in state_backup:
                global_grad_ref[k] = (state_after_server[k].cpu() - state_backup[k]).numpy().astype(np.float32)
            
            # Restore model state for clients (they start from the same point)
            global_model.load_state_dict(state_backup)

            # --- Step B: Client Training (Parallel) ---
            client_tasks = []
            for i in range(n_clients):
                target_gpu = Config.DEVICE_LIST[i % len(Config.DEVICE_LIST)]
                worker_seed = Config.SEED + rnd * 10000 + i
                task = (i, global_state_cpu, partition[i], malicious_flags[i], mal_type, target_gpu, worker_seed, rnd)
                client_tasks.append(task)

            results = []
            desc = f"Rd {rnd + 1}/{Config.ROUNDS}"
            for res in tqdm(pool.imap(client_worker, client_tasks), total=n_clients, desc=desc, leave=False):
                results.append(res)

            updates = [r[0] for r in results]

            # --- Step C: Reputation and Defense Logic (MGGA) ---
            # We pass the 'global_grad_ref' computed in Step A to our new Veto Logic
            
            cluster_ops = compute_cluster_opinions(updates, global_grad_ref, n_clusters=3, rnd=rnd)

            b_list = [op['b'] for op in cluster_ops]
            u_list = [op['u'] for op in cluster_ops]

            # Update reputation (Gamma) using Subjective Logic Fusion
            reputation_cand = np.array([b + Config.NU * u for b, u in zip(b_list, u_list)])

            for i in range(n_clients):
                if reputation[i] >= 0.6: delta_i = 0.9 
                elif reputation[i] >= 0.3: delta_i = 0.6 
                else: delta_i = 0.2 
                # EMA Update
                reputation[i] = delta_i * reputation[i] + (1.0 - delta_i) * reputation_cand[i]

            reputation = np.clip(reputation, 0.0, 1.0)

            # Log Stats
            mal_arr = np.array(malicious_flags)
            logs['avg_rep_mal_log'].append(np.mean(reputation[mal_arr]) if mal_arr.any() else 0.0)
            logs['avg_rep_ben_log'].append(np.mean(reputation[~mal_arr]))
            logs['reputations_log'].append(reputation.copy())

            # --- Step D: Soft-Dropout Aggregation ---
            # This ensures we don't accidentally "zero out" benign clients who are recovering
            
            global_model.eval()
            # Calculate current ASR to penalize known attackers further if needed
            asr_c, asr_t = 0, 0
            with torch.no_grad():
                for xb, yb in loader_asr:
                    xb = xb.to(server_device)
                    out = global_model(xb)
                    asr_c += (out.argmax(1) == 9).sum().item()
                    asr_t += xb.size(0)
            asr_tmp = asr_c / max(1, asr_t)

            reputation_penalized = reputation.copy()
            for i in range(n_clients):
                if malicious_flags[i]:
                    reputation_penalized[i] *= np.exp(-2.0 * asr_tmp)

            alpha = Config.SOFT_DROPOUT_ALPHA
            top_k = max(1, n_clients // 2)
            top_idxs = np.argsort(reputation_penalized)[::-1][:top_k]

            primary = np.zeros(n_clients, dtype=np.float64)
            primary[top_idxs] = reputation_penalized[top_idxs]
            primary_sum = primary.sum()
            if primary_sum > 0:
                primary = (primary / primary_sum) * alpha
            else:
                primary[top_idxs] = alpha / top_k

            u_arr = np.array(u_list, dtype=np.float64)
            inv_u = 1.0 / (u_arr + Config.EPS)
            residual = inv_u / inv_u.sum() * (1.0 - alpha)

            agg_weights = (primary + residual).astype(np.float32)
            agg_weights /= agg_weights.sum()

            # Perform Weighted Aggregation
            local_states = [{k: torch.tensor(v) for k, v in u.items()} for u in updates]
            aggregate_models_weighted(global_model, local_states, agg_weights)

            # --- Step E: Rewards and Fairness ---
            pow_rep = np.power(reputation + Config.EPS, Config.REWARD_Q)
            r_w = 0.7 * (pow_rep / (pow_rep.sum() + Config.EPS))
            f_rep = np.square(reputation + Config.EPS)
            r_r = 0.3 * (f_rep / (f_rep.sum() + Config.EPS))
            penalties = np.array([Config.LAMBDA_PENALTY * max(0, Config.TAU_PENALTY - g) ** 2 for g in reputation])
            net_rewards = r_w + r_r - penalties

            v_centered = reputation - reputation.mean()
            nr_centered = net_rewards - net_rewards.mean()
            rho = (v_centered * nr_centered).sum() / (np.linalg.norm(v_centered) * np.linalg.norm(nr_centered) + Config.EPS)

            if rho < Config.RHO_MIN:
                k = max(1, int(0.2 * n_clients))
                order = np.argsort(net_rewards)
                bottom, top = order[:k], order[-k:]
                transfer = Config.PF_FRACTION * net_rewards[top].sum()
                net_rewards[top] -= ((net_rewards[top] / (net_rewards[top].sum() + Config.EPS)) * transfer)
                net_rewards[bottom] += transfer / k

            logs['rewards_log'].append(net_rewards.copy())
            fair_val = ((net_rewards[net_rewards > 0].sum() ** 2) / (n_clients * (net_rewards ** 2).sum() + Config.EPS))
            logs['fairness_log'].append(fair_val)

            # --- Step F: Final Evaluation ---
            global_model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for xb, yb in DataLoader(test_data, batch_size=256):
                    xb, yb = xb.to(server_device), yb.to(server_device)
                    correct += (global_model(xb).argmax(1) == yb).sum().item()
                    total += yb.size(0)
            acc = correct / total
            logs['acc_log'].append(acc)
            
            asr_c, asr_t = 0, 0
            with torch.no_grad():
                for xb, yb in loader_asr:
                    xb = xb.to(server_device)
                    out = global_model(xb)
                    asr_c += (out.argmax(1) == 9).sum().item()
                    asr_t += xb.size(0)
            asr = asr_c / max(1, asr_t)
            logs['asr_log'].append(asr)

            if logs['convergence_round_log'] is None and acc >= 0.85:
                logs['convergence_round_log'] = rnd + 1

            print(f"Round {rnd + 1} | Acc: {acc:.4f} | ASR: {asr:.4f} | Fair: {fair_val:.3f}")

    for k in logs:
        if isinstance(logs[k], list): logs[k] = np.array(logs[k])
    if logs['convergence_round_log'] is None: logs['convergence_round_log'] = Config.ROUNDS

    return logs
# ==============================================================================
# DATA PARTITIONING AND VISUALIZATION
# ==============================================================================

def get_partitions(dataset, num_clients: int, indices: np.ndarray) -> tuple:
    """Generate IID and Non-IID data partitions for clients.

    Args:
        dataset: Full CIFAR-10 dataset
        num_clients: Number of clients to partition data for
        indices: Indices of client data (excluding server data)

    Returns:
        Tuple of (iid_partitions, noniid_partitions) where each is a list
        of index lists for each client
    """
    # IID Partition: Random shuffle and equal split
    idxs = np.random.permutation(indices)
    iid_parts = [list(p) for p in np.array_split(idxs, num_clients)]

    # Non-IID Partition: Label-skewed (2 shards per client)
    all_targets = np.array(dataset.targets)
    client_targets = all_targets[indices]
    idxs_sorted = indices[np.argsort(client_targets)]

    shards_per_client = 2
    shards = np.array_split(idxs_sorted, num_clients * shards_per_client)
    noniid_parts = [[] for _ in range(num_clients)]
    shard_perm = np.random.permutation(len(shards))
    for i, s in enumerate(shard_perm):
        noniid_parts[i % num_clients].extend(list(shards[s]))

    return iid_parts, noniid_parts


def plot_results(all_results: dict) -> None:
    """Generate and save plots of experiment results.

    Creates visualizations for accuracy, ASR, reputation evolution,
    and fairness metrics.

    Args:
        all_results: Nested dict {tag -> {mal_count -> logs}}
    """
    print("\nGenerating Plots...")

    # 1. Accuracy and ASR Line Plots
    metrics = [
        ('acc', 'Accuracy', 'acc'),
        ('asr', 'Attack Success Rate (0->9)', 'asr'),
    ]

    for key, title, fname in metrics:
        for tag in ['IID', 'NonIID']:
            plt.figure(figsize=(10, 5))
            for m in Config.MALICIOUS_COUNTS:
                if m in all_results[tag]:
                    plt.plot(all_results[tag][m][f'{key}_log'], label=f"mal={m}")
            plt.title(f"{title} ({tag})")
            plt.grid(True)
            plt.legend()
            plt.savefig(
                os.path.join(Config.OUTPUT_DIR, f"{fname}_{tag.lower()}.png")
            )
            plt.close()

    # 2. Final Accuracy Summary
    plt.figure(figsize=(8, 5))
    for tag in ['IID', 'NonIID']:
        finals = [
            all_results[tag][m]['acc_log'][-1] for m in Config.MALICIOUS_COUNTS
        ]
        plt.plot(Config.MALICIOUS_COUNTS, finals, marker='o', label=tag)
    plt.title("Final Accuracy vs Malicious Count")
    plt.xlabel("Number of Malicious Clients")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(Config.OUTPUT_DIR, "final_acc.png"))
    plt.close()

    # 3. Reputation Evolution per Setting
    for tag in ['IID', 'NonIID']:
        for m in Config.MALICIOUS_COUNTS:
            res = all_results[tag][m]
            plt.figure(figsize=(8, 5))
            plt.plot(res['avg_rep_ben_log'], label='Benign', color='green')
            if m > 0:
                plt.plot(
                    res['avg_rep_mal_log'],
                    label='Malicious',
                    color='red',
                    linestyle='--',
                )
            else:
                plt.plot(
                    np.zeros_like(res['avg_rep_ben_log']),
                    label='Malicious (None)',
                    color='gray',
                    linestyle=':',
                    alpha=0.5,
                )

            plt.title(f"Reputation Evolution ({tag}, Mal={m})")
            plt.ylim(-0.05, 1.05)
            plt.ylabel("Average Reputation")
            plt.xlabel("Round")
            plt.grid(True)
            plt.legend()
            plt.savefig(
                os.path.join(Config.OUTPUT_DIR, f"reputation_{tag}_mal{m}.png")
            )
            plt.close()


# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================

if __name__ == '__main__':
    """Main execution entry point for the federated learning experiment."""

    mp.set_start_method('spawn', force=True)

    # Setup Output Directory
    if os.path.exists(Config.OUTPUT_DIR):
        shutil.rmtree(Config.OUTPUT_DIR)
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

    print(
        f"Starting Federated Learning Experiment\n"
        f"  Workers: {Config.MAX_WORKERS}\n"
        f"  Device: {Config.DEVICE_LIST[0]}\n"
        f"  Output: {Config.OUTPUT_DIR}"
    )

    # ==========================================================================
    # Step 1: Load and Prepare Data
    # ==========================================================================
    print("\nLoading CIFAR-10 dataset...")

    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    full_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform_train
    )
    test_data = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_test
    )

    # ==========================================================================
    # Step 2: Create Balanced Server Split
    # ==========================================================================
    print("Splitting data into server and client partitions...")

    server_indices, client_indices = [], []
    targets = np.array(full_dataset.targets)

    for class_idx in np.unique(targets):
        class_indices = np.where(targets == class_idx)[0]
        np.random.shuffle(class_indices)
        server_indices.extend(class_indices[:50])
        client_indices.extend(class_indices[50:])

    server_indices = np.array(server_indices)
    client_indices = np.array(client_indices)
    np.random.shuffle(client_indices)

    print(
        f"Data Allocation: "
        f"Server={len(server_indices)} samples | "
        f"Clients={len(client_indices)} samples"
    )

    # ==========================================================================
    # Step 3: Create Data Loaders
    # ==========================================================================
    loader_server_ref = DataLoader(
        Subset(full_dataset, server_indices), batch_size=64, shuffle=True
    )

    # ASR loader: Test data with label 0 (target class for label-flip attack)
    idx_zeros = [i for i, (x, y) in enumerate(test_data) if y == 0]
    loader_asr = DataLoader(
        Subset(test_data, idx_zeros), batch_size=256, shuffle=False
    )

    # ==========================================================================
    # Step 4: Generate Data Partitions
    # ==========================================================================
    print("Generating IID and Non-IID data partitions...")
    iid_parts, noniid_parts = get_partitions(
        full_dataset, Config.NUM_CLIENTS, client_indices
    )

    # ==========================================================================
    # Step 5: Run Experiments
    # ==========================================================================
    print(f"\nRunning experiments with {len(Config.MALICIOUS_COUNTS)} scenarios...")

    all_results = {'IID': {}, 'NonIID': {}}

    for tag, partition in [('IID', iid_parts), ('NonIID', noniid_parts)]:
        for mal_count in Config.MALICIOUS_COUNTS:
            print(
                f"\n{'='*60}\n"
                f"Experiment: {tag} with {mal_count} Malicious Clients\n"
                f"{'='*60}"
            )
            start_time = time.time()

            # Run experiment
            results = run_experiment(
                partition,
                mal_count,
                test_data,
                loader_asr,
                loader_server_ref,
                Config.MAL_TYPE,
                tag,
            )
            all_results[tag][mal_count] = results

            # Save results to disk
            for metric_name, metric_values in results.items():
                if isinstance(metric_values, np.ndarray):
                    save_path = os.path.join(
                        Config.OUTPUT_DIR,
                        f"{tag}_mal{mal_count}_{metric_name}.npy",
                    )
                    np.save(save_path, metric_values)

            elapsed = time.time() - start_time
            print(f"Completed in {elapsed:.2f} seconds")

    # ==========================================================================
    # Step 6: Generate Visualizations
    # ==========================================================================
    plot_results(all_results)

    print("\n" + "=" * 60)
    print("All experiments completed successfully!")
    print(f"Results saved to: {Config.OUTPUT_DIR}")
    print("=" * 60)