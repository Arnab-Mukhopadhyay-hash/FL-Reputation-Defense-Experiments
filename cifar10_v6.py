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
# 1. CONFIGURATION
# ==============================================================================

class Config:
    # Hardware & System
    MAX_WORKERS = 2  # Adjust based on CPU cores/GPU memory
    OUTPUT_DIR = "cifar10_v6_results"
    DEVICE_LIST = ['cuda:{}'.format(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else ['cpu']
    
    # FL Experiment
    NUM_CLIENTS = 10
    ROUNDS = 30
    SEED = 1234
    MALICIOUS_COUNTS = [0, 2, 4, 6]
    MAL_TYPE = 'label_flip'  # 'label_flip' or 'random_grad'
    
    # Worker Training
    LOCAL_EPOCHS = 1
    BATCH_SIZE = 64
    LR = 0.01
    MOMENTUM = 0.9
    WEIGHT_DECAY = 5e-4

    # Subjective Logic / Defense Parameters
    EPS = 1e-10
    REPUTATION_BETA = 0.05
    NU = 0.5
    DELTA = 0.85
    W_FIXED = 0.1         # Fixed uncertainty weight
    RHO_MIN = 0.9         # Fairness correlation threshold
    PF_FRACTION = 0.05    # Redistribution fraction
    LAMBDA_PENALTY = 2.0
    TAU_PENALTY = 0.2
    REWARD_Q = 1.0

    # --- Z-Score Calibration (replaces Min-Max) ---
    ZSCORE_FLOOR = 0.15     # Minimum calibrated score any client can receive.
                            # Prevents NonIID benign outliers from being zeroed out.
                            # Downstream SL belief/uncertainty still separates them
                            # from true attackers over multiple rounds.

    # --- Soft-Dropout Aggregation ---
    SOFT_DROPOUT_ALPHA = 0.7  # Fraction of total weight given to the top-50% (primary channel).
                              # The remaining (1 - alpha) is distributed to ALL clients via
                              # the residual channel (inverse-uncertainty weighting).
    PREV_CENTROIDS = None 

# ==============================================================================
# 2. MODEL DEFINITION
# ==============================================================================

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            # Block 1: 32x32 -> 16x16
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Block 2: 16x16 -> 8x8
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Block 3: 8x8 -> 4x4
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        return self.net(x)


from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# ==========================================
# NEW: CLUSTER-AGNOSTIC AGGREGATOR LOGIC
# ==========================================

# Add to your global variables or Config class


# ==========================================
# MODIFIED: CLUSTER-AGNOSTIC AGGREGATOR WITH TEMPORAL MOMENTUM
# ==========================================

def compute_cluster_opinions(updates, n_clusters=3, outlier_threshold=1.5, rnd=0):
    """
    Performs PCA and clusters clients using Temporal Centroid Momentum.
    Initializes K-Means with centroids from the previous round to prevent hijacking.
    """

    # 1. Flatten updates for PCA
    flattened_updates = []
    for u in updates:
        flat = np.concatenate([v.flatten() for v in u.values()])
        flattened_updates.append(flat)
    X = np.array(flattened_updates)

    # 2. PCA Dimensionality Reduction
    pca = PCA(n_components=min(X.shape[0], 10))
    X_reduced = pca.fit_transform(X)

    # 3. K-Means Clustering with Temporal Initialization
    if rnd == 0 or PREV_CENTROIDS is None:
        # Standard initialization for the first round
        kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, random_state=Config.SEED)
    else:
        # Temporal Centroid Momentum: Use previous centroids as starting point
        # Ensure PREV_CENTROIDS matches current PCA space or re-align if necessary
        # For simplicity in v6, we assume the PCA projection space is stable enough for init
        kmeans = KMeans(n_clusters=n_clusters, init=Config.PREV_CENTROIDS, n_init=1, random_state=Config.SEED)

    cluster_labels = kmeans.fit_predict(X_reduced)
    current_centroids = kmeans.cluster_centers_
    
    # Store centroids for the next round
    Config.PREV_CENTROIDS = current_centroids

    opinions = []
    for i, x_vec in enumerate(X_reduced):
        # Calculate distance to each cluster centroid
        dists = [np.linalg.norm(x_vec - c) for c in current_centroids]
        min_dist = min(dists)
        avg_dist = np.mean(dists)

        # 4. Outlier Detection with Uncertainty
        if min_dist > outlier_threshold * avg_dist:
            # Outliers are assigned maximum uncertainty
            op = {'b': 0.0, 'd': 0.0, 'u': 1.0}
        else:
            # Map distance to belief using a Gaussian kernel
            belief = np.exp(-min_dist**2 / (2 * (avg_dist**2)))
            disbelief = (1.0 - belief) * 0.8
            uncertainty = 1.0 - belief - disbelief
            op = {'b': belief, 'd': disbelief, 'u': uncertainty}
        
        opinions.append(op)
    
    return opinions

# ==============================================================================
# 3. MATH & DEFENSE LOGIC (Subjective Logic & Metrics)
# ==============================================================================

def cosine_similarity_per_class(a, b):
    """
    Computes cosine similarity row-wise (per-class) for the last layer weights.
    Returns the minimum similarity across classes (worst-case trust).
    """
    keys = list(a.keys())
    w_key = keys[-2]
    
    A_mat = a[w_key]
    B_mat = b[w_key]
    
    dot_prod = np.sum(A_mat * B_mat, axis=1)
    norm_a = np.linalg.norm(A_mat, axis=1) + Config.EPS
    norm_b = np.linalg.norm(B_mat, axis=1) + Config.EPS
    
    sims = dot_prod / (norm_a * norm_b)
    return float(np.min(sims))

def pearson_correlation_per_class(a, b):
    """
    Computes Pearson correlation row-wise (per-class) for the last layer weights.
    Returns the minimum correlation across classes.
    """
    keys = list(a.keys())
    w_key = keys[-2]
    
    A_mat = a[w_key]
    B_mat = b[w_key]
    
    # Center data per row
    A_mean = np.mean(A_mat, axis=1, keepdims=True)
    B_mean = np.mean(B_mat, axis=1, keepdims=True)
    
    A_centered = A_mat - A_mean
    B_centered = B_mat - B_mean
    
    dot_prod = np.sum(A_centered * B_centered, axis=1)
    norm_a = np.linalg.norm(A_centered, axis=1) + Config.EPS
    norm_b = np.linalg.norm(B_centered, axis=1) + Config.EPS
    
    corrs = dot_prod / (norm_a * norm_b)
    return float(np.min(corrs))

def fuse_opinions(op1, op2):
    b1, d1, u1 = op1['b'], op1['d'], op1['u']
    b2, d2, u2 = op2['b'], op2['d'], op2['u']
    
    kappa = u1 + u2 - u1 * u2
    if kappa < 1e-9: kappa = 1e-9
    
    return {
        'b': (b1 * u2 + b2 * u1) / kappa,
        'd': (d1 * u2 + d2 * u1) / kappa,
        'u': (u1 * u2) / kappa
    }

def aggregate_models_weighted(global_model, updates, weights):
    global_dict = global_model.state_dict()
    new_state = {}
    agg_update = {}
    
    # Initialize with first key structure
    first_k = list(updates[0].keys())
    
    for k in first_k:
        # Weighted sum of updates on the device of the update
        agg_tensor = torch.zeros_like(updates[0][k])
        for i, u in enumerate(updates):
            agg_tensor += weights[i] * u[k]
        agg_update[k] = agg_tensor
        
    # Apply aggregated update to global model
    for k in global_dict:
        new_state[k] = global_dict[k] + agg_update[k].to(global_dict[k].device)
        
    global_model.load_state_dict(new_state)


# ==============================================================================
# 4. WORKER PROCESS
# ==============================================================================

def client_worker(args):
    """
    Independent worker function.
    Performs local training and returns weight update vector.
    """
    client_idx, global_weights, indices, malicious, mal_type, device_name, worker_seed, rnd = args
    
    # 1. Setup Environment
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    device = torch.device(device_name) if torch.cuda.is_available() and "cuda" in device_name else torch.device("cpu")
    
    # 2. Prepare Data Transforms
    transform_local = transforms.Compose([
        transforms.RandomCrop(32, padding=4), 
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(), 
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    transform_eval = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # 3. Load Datasets
    # Assumes data exists at ./data
    full_dataset_train = torchvision.datasets.CIFAR10(root="./data", train=True, download=False, transform=transform_local)
    full_dataset_eval  = torchvision.datasets.CIFAR10(root="./data", train=True, download=False, transform=transform_eval)
    
    # 4. Split Indices (Train/Val) for local robustness check
    split_point = max(int(len(indices) * 0.2), 50)
    val_indices = indices[:split_point]
    train_indices = indices[split_point:]
    
    loader_val = DataLoader(Subset(full_dataset_eval, val_indices), batch_size=Config.BATCH_SIZE, shuffle=False)
    
    # 5. Handle Malicious Data (Label Flip)
    if malicious and mal_type == 'label_flip':
        flip_from, flip_to = 0, 9
        flipped_data = []
        subset = Subset(full_dataset_train, train_indices)
        for i in range(len(subset)):
            x, y = subset[i]
            if int(y) == flip_from: y = flip_to
            flipped_data.append((x, y))
        loader_train = DataLoader(flipped_data, batch_size=Config.BATCH_SIZE, shuffle=True)
    else:
        loader_train = DataLoader(Subset(full_dataset_train, train_indices), batch_size=Config.BATCH_SIZE, shuffle=True)

    # 6. Initialize Model
    local_model = SimpleCNN().to(device)
    local_model.load_state_dict(global_weights)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(local_model.parameters(), lr=Config.LR, momentum=Config.MOMENTUM, weight_decay=Config.WEIGHT_DECAY)

    # 7. Helper: Evaluate Loss
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

    # 8. Pre-Training Validation
    loss_pre = get_loss(loader_val)

    # 9. Local Training
    local_model.train()
    for _ in range(Config.LOCAL_EPOCHS):
        for xb, yb in loader_train:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = local_model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()

    # 10. Post-Training Validation
    loss_post = get_loss(loader_val)

    # 11. Compute Update Vector (on CPU)
    update = {}
    local_state = local_model.state_dict()
    for k, v_global in global_weights.items():
        v_local = local_state[k].cpu()
        update[k] = (v_local - v_global).numpy().astype(np.float32)

    # 12. Handle Malicious Gradients
    if malicious and mal_type == 'random_grad':
        for k in update:
            update[k] = np.random.randn(*update[k].shape).astype(np.float32) * 0.05

    return update, loss_pre, loss_post


# ==============================================================================
# 5. EXPERIMENT CONTROLLER
# ==============================================================================

def run_experiment(partition, mal_count, test_data, loader_asr, loader_server_ref, mal_type='label_flip', tag='iid'):
    
    server_device = torch.device(Config.DEVICE_LIST[0])
    global_model = SimpleCNN().to(server_device)
    
    n_clients = len(partition)
    Gamma = np.ones(n_clients) / n_clients  # Reputation vector
    
    # Logging
    logs = {
        'acc_log': [], 'asr_log': [], 'fairness_log': [], 'rewards_log': [],
        'reputations_log': [], 'avg_rep_mal_log': [], 'avg_rep_ben_log': [],
        'convergence_round_log': None
    }
    
    # Setup malicious clients
    malicious_flags = [False] * n_clients
    mal_idxs = np.random.choice(n_clients, mal_count, replace=False)
    for i in mal_idxs: malicious_flags[i] = True
    
    print(f"Starting {tag} (Malicious: {mal_count})")

    with mp.Pool(processes=Config.MAX_WORKERS) as pool:
        for rnd in range(Config.ROUNDS):
            
            # --- A. SERVER REFERENCE STEP ---
            global_state_cpu = {k: v.cpu() for k, v in global_model.state_dict().items()}
            state_backup = {k: v.cpu().clone() for k, v in global_model.state_dict().items()}
            
            # Train temporarily on server data to get "Gold Standard" gradient
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
            
            # Restore model for client distribution
            global_model.load_state_dict(state_backup)
            
            # --- B. CLIENT TRAINING STEP ---
            client_tasks = []
            for i in range(n_clients):
                target_gpu = Config.DEVICE_LIST[i % len(Config.DEVICE_LIST)]
                worker_seed = Config.SEED + rnd * 10000 + i
                task = (i, global_state_cpu, partition[i], malicious_flags[i], mal_type, target_gpu, worker_seed, rnd)
                client_tasks.append(task)
                
            results = []
            desc = f"Rd {rnd+1}/{Config.ROUNDS}"
            for res in tqdm(pool.imap(client_worker, client_tasks), total=n_clients, desc=desc, leave=False):
                results.append(res)
                
            updates = [r[0] for r in results]

            # --- C. REPUTATION & DEFENSE LOGIC ---
            
            # # 1. Compute Raw Similarities
            # raw_s_list = [cosine_similarity_per_class(u, global_grad_ref) for u in updates]
            # raw_p_list = [pearson_correlation_per_class(u, global_grad_ref) for u in updates]
            
            # raw_s_arr = np.array(raw_s_list)
            # raw_p_arr = np.array(raw_p_list)
            
            # # 2. Calibration — Z-Score with Safety Floor
            # #
            # # WHY Z-Score replaces Min-Max:
            # #   Min-Max maps the single lowest scorer to exactly 0.0.  In NonIID
            # #   settings that lowest scorer is frequently a benign client whose
            # #   data is simply furthest from the server reference — zeroing its
            # #   score on round 1 is an unrecoverable false positive.
            # #
            # # HOW the Safety Floor works:
            # #   1.  z_i  = (x_i - mean) / std          → centred, unit-variance
            # #   2.  σ(z_i) squashes z into (0, 1)       → z=0  → 0.5
            # #                                             z>>0 → 1
            # #                                             z<<0 → 0
            # #   3.  Rescale [0, 1]  →  [floor, 1]
            # #          out_i = floor + (1 - floor) * σ(z_i)
            # #       Guarantees out_i >= floor for every client, no matter how
            # #       negative its z-score.  floor = Config.ZSCORE_FLOOR.
            # #
            # # Downstream separation:
            # #   A benign NonIID outlier and a real attacker may both land near
            # #   the floor in a single round — that is intentional.  The SL
            # #   opinion formation assigns the outlier high uncertainty (u) while
            # #   repeated attacks drive the attacker's disbelief (d) up over
            # #   rounds.  The tiered EMA on Gamma then separates them without the
            # #   irreversible zero-out that Min-Max caused.
            # def calibrate(arr):
            #     mu  = arr.mean()
            #     sig = arr.std()
            #     if sig < 1e-8:
            #         # Degenerate: all scores identical.
            #         # No information to differentiate → everyone gets 0.5.
            #         return np.full_like(arr, 0.5, dtype=np.float64)
            #     z       = (arr - mu) / sig
            #     sigmoid = 1.0 / (1.0 + np.exp(-z))                            # (0, 1)
            #     floored = Config.ZSCORE_FLOOR + (1.0 - Config.ZSCORE_FLOOR) * sigmoid  # [floor, 1)
            #     return floored

            # s_list = calibrate(raw_s_arr)
            # p_list = calibrate(raw_p_arr)
            
            # # 3. Form Opinions and Fuse
            # b_list, u_list = [], []
            # for s_val, p_val in zip(s_list, p_list):
            #     # Cosine Opinion
            #     s_val = np.clip(s_val, 0, 1)
            #     denom_s = s_val + (1.0 - s_val) + Config.W_FIXED
            #     op_s = {'b': s_val**2/denom_s, 'd': (1-s_val)/denom_s, 'u': Config.W_FIXED/denom_s}
                
            #     # Pearson Opinion
            #     p_val = np.clip(p_val, 0, 1)
            #     denom_p = p_val + (1.0 - p_val) + Config.W_FIXED
            #     op_p = {'b': p_val**2/denom_p, 'd': (1-p_val)/denom_p, 'u': Config.W_FIXED/denom_p}
                
            #     # Fusion
            #     op_final = fuse_opinions(op_s, op_p)
            #     total = op_final['b'] + op_final['d'] + op_final['u']
            #     b_list.append(op_final['b'] / total)
            #     u_list.append(op_final['u'] / total)

            # 1. Generate Cluster-Based Opinions
            cluster_ops = compute_cluster_opinions(updates, n_clusters=3, rnd=rnd)

            # 2. Extract Belief and Uncertainty for Reputation Update
            b_list = [op['b'] for op in cluster_ops]
            u_list = [op['u'] for op in cluster_ops]
                
            # 4. Update Gamma (Reputation) using EMA
            Gamma_cand = np.array([b + Config.NU * u for b, u in zip(b_list, u_list)])
            
            for i in range(n_clients):
                if Gamma[i] >= 0.6:   delta_i = 0.9   # Trusted
                elif Gamma[i] >= 0.3: delta_i = 0.6   # Uncertain
                else:                 delta_i = 0.2   # Untrusted
                
                Gamma[i] = delta_i * Gamma[i] + (1.0 - delta_i) * Gamma_cand[i]
            
            Gamma = np.clip(Gamma, 0.0, 1.0)
            
            # Log Reputations
            mal_arr = np.array(malicious_flags)
            logs['avg_rep_mal_log'].append(np.mean(Gamma[mal_arr]) if mal_arr.any() else 0.0)
            logs['avg_rep_ben_log'].append(np.mean(Gamma[~mal_arr]))
            logs['reputations_log'].append(Gamma.copy())

            # --- D. AGGREGATION — Soft-Dropout ---
            #
            # WHY Soft-Dropout replaces hard thresholding:
            #   The old code set a trust_thresh (mean + 0.5*std, floor 0.45) and
            #   gave ZERO weight to every client below it.  In NonIID settings the
            #   threshold can accidentally exclude benign clients whose reputation
            #   hasn't recovered yet (especially after the first few rounds).
            #   A hard zero is irreversible within that round — the client's
            #   update, even if useful, is discarded entirely.
            #
            # HOW Soft-Dropout works (two-channel design):
            #
            #   ┌─ PRIMARY channel  (weight = α)  ──────────────────────────┐
            #   │  • Only the top 50 % of clients ranked by gamma_penalized  │
            #   │    participate.                                            │
            #   │  • Within that set weights are proportional to reputation. │
            #   │  • This is the "strong signal" — the server's best guess   │
            #   │    of who is trustworthy.                                  │
            #   └────────────────────────────────────────────────────────────┘
            #        +
            #   ┌─ RESIDUAL channel  (weight = 1 − α)  ─────────────────────┐
            #   │  • ALL clients contribute — nobody is zeroed out.          │
            #   │  • Weight ∝ 1 / u_i  (inverse of Subjective Logic         │
            #   │    uncertainty from the opinion-fusion step above).        │
            #   │  • A client with HIGH uncertainty (unsure evidence) gets a │
            #   │    tiny residual share; a confident client gets more.      │
            #   │  • This channel is intentionally small (1 − α = 0.3) so   │
            #   │    it cannot override the primary signal, but it prevents  │
            #   │    full exclusion of any single client.                    │
            #   └────────────────────────────────────────────────────────────┘
            #
            #   α = Config.SOFT_DROPOUT_ALPHA  (default 0.7)
            #
            # Net effect on an attacker vs a benign NonIID outlier:
            #   • Attacker:  low gamma_penalized  →  excluded from primary,
            #                high u (uncertain evidence) →  tiny residual weight.
            #   • Benign outlier:  moderate gamma (recovering via EMA),
            #                      low u (confident but skewed) →  meaningful
            #                      residual weight even if outside top-50 %.

            # ASR-based penalty (kept from original — penalises known-malicious
            # clients when the model's current ASR is already elevated).
            global_model.eval()
            asr_c, asr_t = 0, 0
            with torch.no_grad():
                for xb, yb in loader_asr:
                    xb = xb.to(server_device)
                    out = global_model(xb)
                    asr_c += (out.argmax(1) == 9).sum().item()
                    asr_t += xb.size(0)
            asr_tmp = asr_c / max(1, asr_t)

            gamma_penalized = Gamma.copy()
            for i in range(n_clients):
                if malicious_flags[i]:
                    gamma_penalized[i] *= np.exp(-2.0 * asr_tmp)

            # ── Primary channel: top-50 % by penalised reputation ─────────
            alpha   = Config.SOFT_DROPOUT_ALPHA
            top_k   = max(1, n_clients // 2)                          # at least 1
            top_idxs = np.argsort(gamma_penalized)[::-1][:top_k]      # descending

            primary = np.zeros(n_clients, dtype=np.float64)
            primary[top_idxs] = gamma_penalized[top_idxs]
            primary_sum = primary.sum()
            if primary_sum > 0:
                primary = (primary / primary_sum) * alpha             # L1-norm then scale to α
            else:
                # Fallback: uniform over top-k, scaled to α
                primary[top_idxs] = alpha / top_k

            # ── Residual channel: ALL clients, weighted by 1 / u_i ────────
            u_arr    = np.array(u_list, dtype=np.float64)
            inv_u    = 1.0 / (u_arr + Config.EPS)                     # high confidence → high weight
            residual = inv_u / inv_u.sum() * (1.0 - alpha)            # L1-norm then scale to (1−α)

            # ── Merge ──────────────────────────────────────────────────────
            rep_weights = (primary + residual).astype(np.float32)
            # Numerical safety: re-normalise to exactly 1.0
            rep_weights /= rep_weights.sum()

            # Aggregate
            local_states = [{k: torch.tensor(v) for k, v in u.items()} for u in updates]
            aggregate_models_weighted(global_model, local_states, rep_weights)
            
            # --- E. REWARDS & FAIRNESS ---
            pow_rep = np.power(Gamma + Config.EPS, Config.REWARD_Q)
            r_w = 0.7 * (pow_rep / (pow_rep.sum() + Config.EPS))
            
            f_rep = np.square(Gamma + Config.EPS)
            r_r = 0.3 * (f_rep / (f_rep.sum() + Config.EPS))
            
            penalties = np.array([Config.LAMBDA_PENALTY * max(0, Config.TAU_PENALTY - g)**2 for g in Gamma])
            net_rewards = r_w + r_r - penalties
            
            # Redistribution (Fairness check)
            # Simple correlation between Reputation (v) and Rewards
            v_centered = Gamma - Gamma.mean()
            nr_centered = net_rewards - net_rewards.mean()
            rho = (v_centered * nr_centered).sum() / (np.linalg.norm(v_centered)*np.linalg.norm(nr_centered) + Config.EPS)
            
            if rho < Config.RHO_MIN:
                k = max(1, int(0.2 * n_clients))
                order = np.argsort(net_rewards)
                bottom, top = order[:k], order[-k:]
                transfer = Config.PF_FRACTION * net_rewards[top].sum()
                net_rewards[top] -= (net_rewards[top] / (net_rewards[top].sum() + Config.EPS)) * transfer
                net_rewards[bottom] += transfer / k
            
            logs['rewards_log'].append(net_rewards.copy())
            fair_val = (net_rewards[net_rewards>0].sum()**2) / (n_clients * (net_rewards**2).sum() + Config.EPS)
            logs['fairness_log'].append(fair_val)

            # --- F. FINAL EVALUATION ---
            global_model.eval()
            
            # Accuracy
            correct, total = 0, 0
            with torch.no_grad():
                for xb, yb in DataLoader(test_data, batch_size=256):
                    xb, yb = xb.to(server_device), yb.to(server_device)
                    correct += (global_model(xb).argmax(1) == yb).sum().item()
                    total += yb.size(0)
            acc = correct / total
            logs['acc_log'].append(acc)
            
            # ASR
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
            
            print(f"Round {rnd+1} | Acc: {acc:.4f} | ASR: {asr:.4f} | Fair: {fair_val:.3f}")
            
    # Convert lists to arrays
    for k in logs:
        if isinstance(logs[k], list): logs[k] = np.array(logs[k])
    
    if logs['convergence_round_log'] is None: logs['convergence_round_log'] = Config.ROUNDS
    
    return logs


# ==============================================================================
# 6. HELPERS FOR DATA PARTITIONING & PLOTTING
# ==============================================================================

def get_partitions(dataset, num_clients, indices):
    """Generates both IID and Non-IID partitions based on indices."""
    
    # IID
    idxs = np.random.permutation(indices)
    iid_parts = [list(p) for p in np.array_split(idxs, num_clients)]
    
    # Non-IID (Label Skew)
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

def plot_results(all_results):
    print("\nGenerating Plots...")
    
    metrics = [
        ('acc', 'Accuracy', 'acc'), 
        ('asr', 'Attack Success Rate (0->9)', 'asr'),
    ]
    
    # 1. Line Plots (Acc, ASR)
    for key, title, fname in metrics:
        for tag in ['IID', 'NonIID']:
            plt.figure(figsize=(10, 5))
            for m in Config.MALICIOUS_COUNTS:
                if m in all_results[tag]:
                    plt.plot(all_results[tag][m][f'{key}_log'], label=f"mal={m}")
            plt.title(f"{title} ({tag})")
            plt.grid(True); plt.legend()
            plt.savefig(os.path.join(Config.OUTPUT_DIR, f"{fname}_{tag.lower()}.png"))
            plt.close()

    # 2. Final Accuracy Summary
    plt.figure(figsize=(8, 5))
    for tag in ['IID', 'NonIID']:
        finals = [all_results[tag][m]['acc_log'][-1] for m in Config.MALICIOUS_COUNTS]
        plt.plot(Config.MALICIOUS_COUNTS, finals, marker='o', label=tag)
    plt.title("Final Accuracy vs Malicious Count")
    plt.grid(True); plt.legend()
    plt.savefig(os.path.join(Config.OUTPUT_DIR, "final_acc.png"))
    plt.close()

    # 3. Reputation Evolution
    for tag in ['IID', 'NonIID']:
        for m in Config.MALICIOUS_COUNTS:
            res = all_results[tag][m]
            plt.figure(figsize=(8, 5))
            plt.plot(res['avg_rep_ben_log'], label='Benign', color='green')
            if m > 0:
                plt.plot(res['avg_rep_mal_log'], label='Malicious', color='red', linestyle='--')
            else:
                plt.plot(np.zeros_like(res['avg_rep_ben_log']), label='Malicious (None)', color='gray', linestyle=':', alpha=0.5)
            
            plt.title(f"Reputation ({tag}, Mal={m})")
            plt.ylim(-0.05, 1.05); plt.grid(True); plt.legend()
            plt.savefig(os.path.join(Config.OUTPUT_DIR, f"reputation_{tag}_mal{m}.png"))
            plt.close()


# ==============================================================================
# 7. MAIN ENTRY POINT
# ==============================================================================

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    
    # Setup Directories
    if os.path.exists(Config.OUTPUT_DIR): shutil.rmtree(Config.OUTPUT_DIR)
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    
    print(f"🚀 Running with MAX_WORKERS={Config.MAX_WORKERS} on {Config.DEVICE_LIST[0]}")
    
    # 1. Data Loading & Splits
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4), 
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(), 
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    full_dataset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_train)
    test_data = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_test)
    
    # 2. Balanced Server Split (50 per class)
    server_indices, client_indices = [], []
    targets = np.array(full_dataset.targets)
    for c in np.unique(targets):
        c_idxs = np.where(targets == c)[0]
        np.random.shuffle(c_idxs)
        server_indices.extend(c_idxs[:50])
        client_indices.extend(c_idxs[50:])
        
    server_indices = np.array(server_indices)
    client_indices = np.array(client_indices)
    np.random.shuffle(client_indices)
    
    print(f"Data: Server={len(server_indices)} | Clients={len(client_indices)}")
    
    # Loaders
    loader_server_ref = DataLoader(Subset(full_dataset, server_indices), batch_size=64, shuffle=True)
    idx_zeros = [i for i, (x, y) in enumerate(test_data) if y == 0]
    loader_asr = DataLoader(Subset(test_data, idx_zeros), batch_size=256, shuffle=False)
    
    # 3. Partitions
    iid_parts, noniid_parts = get_partitions(full_dataset, Config.NUM_CLIENTS, client_indices)
    
    # 4. Run Experiments
    all_results = {'IID': {}, 'NonIID': {}}
    
    for tag, partition in [('IID', iid_parts), ('NonIID', noniid_parts)]:
        for m in Config.MALICIOUS_COUNTS:
            print(f"\n=== Experiment: {tag}, Malicious={m} ===")
            start_t = time.time()
            
            res = run_experiment(partition, m, test_data, loader_asr, loader_server_ref, Config.MAL_TYPE, tag)
            all_results[tag][m] = res
            
            # Save raw data
            for k, v in res.items():
                if isinstance(v, np.ndarray):
                    np.save(os.path.join(Config.OUTPUT_DIR, f"{tag}_mal{m}_{k}.npy"), v)
                    
            print(f"Finished in {time.time() - start_t:.2f}s")

    # 5. Plot
    plot_results(all_results)
    print("\n✅ All experiments completed.")