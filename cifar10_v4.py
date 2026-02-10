import os
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

# ======================================================
# 1. CONFIGURATION
# ======================================================
class Config:
    NUM_CLIENTS = 10
    ROUNDS = 30
    LOCAL_EPOCHS = 1
    BATCH_SIZE = 64
    LR = 0.01
    MOMENTUM = 0.9
    WEIGHT_DECAY = 5e-4
    
    # Use CUDA if available, else CPU
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    SEED = 1234
    
    # Defense Parameters
    REP_EMA = 0.8  # Exponential Moving Average for Reputation
    EPS = 1e-10    # Epsilon for numerical stability
    
    # Experiment Settings
    MAL_COUNTS = [0, 2, 4, 6]  # Number of malicious clients to test
    CONV_ACC = 0.85            # Target accuracy for convergence check
    NUM_WORKERS = 2            # Number of parallel processes


# ======================================================
# 2. MODEL DEFINITION
# ======================================================
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.net(x)


# ======================================================
# 3. UTILITIES & DEFENSE MATH
# ======================================================
def flatten_params(model_state):
    """Flattens a state dict into a single 1D numpy array."""
    # FIX: Removed .cpu().numpy() because the input is already a numpy array
    return np.concatenate([v.flatten() for v in model_state.values()])

def cosine_similarity(a, b):
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return np.dot(a, b) / (norm_a * norm_b + Config.EPS)

def centered_kernel_alignment(a, b):
    """
    Computes Centered Kernel Alignment (Linear CKA) between two vectors.
    Mathematically equivalent to Pearson Correlation for 1D vectors.
    
    Formula: <a - mean(a), b - mean(b)> / (||a-mean(a)|| * ||b-mean(b)||)
    
    Why this instead of raw CKA? 
    Raw CKA (squared dot product) loses the sign (direction). 
    We need to preserve the sign so that malicious updates (opposite direction) 
    get negative scores, not positive ones.
    """
    # 1. Center the vectors (Kernel Centering)
    a_centered = a - np.mean(a)
    b_centered = b - np.mean(b)
    
    # 2. Compute Cosine of centered vectors (Pearson Correlation)
    norm_a = np.linalg.norm(a_centered)
    norm_b = np.linalg.norm(b_centered)
    
    cka_score = np.dot(a_centered, b_centered) / (norm_a * norm_b + Config.EPS)
    return cka_score

def get_pca_direction(updates_matrix):
    """
    Computes the Principal Component of the updates using SVD.
    Input: updates_matrix (n_clients, n_params)
    Output: The first principal component vector (n_params,)
    Using SVD is efficient: We don't need the massive (n_params x n_params) covariance matrix.
    """
    # 1. Center the data
    centered = updates_matrix - updates_matrix.mean(axis=0, keepdims=True)
    
    # 2. Compute SVD. 
    # specific shape: (n_clients, n_params) -> U, S, Vt
    # We only need the first row of Vt (which is the first eigenvector of the covariance)
    # Using full_matrices=False prevents creating massive matrices
    try:
        _, _, Vt = np.linalg.svd(centered, full_matrices=False)
        return Vt[0] # The principal direction
    except np.linalg.LinAlgError:
        # Fallback if SVD fails (rare)
        return np.ones(updates_matrix.shape[1])

def evaluate_model(model, loader):
    """Returns accuracy (0-1)."""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(Config.DEVICE), y.to(Config.DEVICE)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / max(1, total)

def evaluate_asr(model, loader_zeros):
    """Returns Attack Success Rate (fraction of '0' images classified as '9')."""
    model.eval()
    hit, total = 0, 0
    with torch.no_grad():
        for x, y in loader_zeros:
            x, y = x.to(Config.DEVICE), y.to(Config.DEVICE)
            # We only care about images that are actually label 0
            # (Though loader_zeros should already only contain 0s)
            mask = (y == 0)
            if mask.sum() == 0: continue
            
            preds = model(x[mask]).argmax(dim=1)
            hit += (preds == 9).sum().item() # Target label is 9
            total += mask.sum().item()
    return hit / max(1, total)


# ======================================================
# 4. CLIENT WORKER
# ======================================================
def client_update(args):
    """
    Runs local training for a single client.
    Args: (client_id, model_state_cpu, indices, is_malicious)
    """
    cid, global_state, indices, malicious = args

    # Set seeds for reproducibility
    torch.manual_seed(Config.SEED + cid)
    np.random.seed(Config.SEED + cid)

    # Load Model
    model = SimpleCNN().to(Config.DEVICE)
    model.load_state_dict(global_state)
    model.train()

    # Load Data (Re-instantiating dataset is safer for multiprocessing than passing it)
    transform = transforms.ToTensor()
    dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
    
    # Create Local Dataset (Subset)
    data_list = []
    for idx in indices:
        x, y = dataset[idx]
        # Malicious Behavior: Label Flip Attack (0 -> 9)
        if malicious and y == 0:
            y = 9
        data_list.append((x, y))

    loader = DataLoader(data_list, batch_size=Config.BATCH_SIZE, shuffle=True)
    optimizer = optim.SGD(model.parameters(), lr=Config.LR, momentum=Config.MOMENTUM, weight_decay=Config.WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    # Local Training Loop
    for _ in range(Config.LOCAL_EPOCHS):
        for x, y in loader:
            x, y = x.to(Config.DEVICE), y.to(Config.DEVICE)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

    # Compute Update (Weight Difference)
    # Update = New_Weights - Global_Weights
    local_state = model.state_dict()
    update_dict = {}
    for k, v in local_state.items():
        update_dict[k] = (v.cpu() - global_state[k]).numpy()

    return update_dict


# ======================================================
# 5. FEDERATED LEARNING LOOP
# ======================================================
def run_experiment(partitions, malicious_flags, test_loader, zero_loader):
    # Initialize Global Model
    model = SimpleCNN().to(Config.DEVICE)
    
    # Initialize Reputation (Gamma)
    # Everyone starts with equal trust (1/N)
    Gamma = np.ones(Config.NUM_CLIENTS) / Config.NUM_CLIENTS

    # Logs
    history = {
        'acc': [],
        'asr': [],
        'rep_ben': [],
        'rep_mal': [],
        'conv_round': Config.ROUNDS
    }
    
    print(f"Starting Experiment: Malicious Clients = {sum(malicious_flags)}")

    for rnd in range(Config.ROUNDS):
        # Prepare Global State for Workers (Must be on CPU for pickling)
        global_state_cpu = {k: v.cpu() for k, v in model.state_dict().items()}

        # Prepare Tasks
        tasks = [
            (i, global_state_cpu, partitions[i], malicious_flags[i]) 
            for i in range(Config.NUM_CLIENTS)
        ]

        # Parallel Training
        with mp.Pool(Config.NUM_WORKERS) as pool:
            updates_dicts = pool.map(client_update, tasks)

        # ----------------------------------------
        # DEFENSE & AGGREGATION LOGIC
        # ----------------------------------------
        
        # 1. Flatten updates for analysis
        # Shape: (n_clients, n_parameters)
        flat_updates = np.stack([flatten_params(u) for u in updates_dicts])
        
        # 2. PCA Defense: Find the "Mainstream" direction
        principal_dir = get_pca_direction(flat_updates)

        # 3. Calculate Scores using Centered Kernel Alignment (CKA)
        # We calculate CKA between each client's update and the PCA principal direction.
        # We clamp at 0 because negative correlation means "opposite direction" (likely malicious)
        scores = np.array([max(0.0, centered_kernel_alignment(u, principal_dir)) for u in flat_updates])
        
        # 4. Normalize Scores (Subjective Logic "Belief")
        # We scale by update magnitude to filter out lazy clients (optional but good)
        magnitudes = np.linalg.norm(flat_updates, axis=1)
        magnitudes /= (magnitudes.max() + Config.EPS)
        
        belief = scores * magnitudes
        belief /= (belief.sum() + Config.EPS) # Normalize to sum to 1

        # 5. Update Reputation (Gamma) using EMA
        Gamma = Config.REP_EMA * Gamma + (1 - Config.REP_EMA) * belief
        Gamma /= (Gamma.sum() + Config.EPS) # Renormalize

        # 6. Weighted Aggregation
        # New_Weight = Old_Weight + Sum(Reputation * Update)
        new_state = {}
        first_key = list(model.state_dict().keys())[0]
        
        for k in global_state_cpu:
            # Weighted sum of updates
            weighted_update = np.zeros_like(updates_dicts[0][k])
            for i in range(Config.NUM_CLIENTS):
                weighted_update += Gamma[i] * updates_dicts[i][k]
            
            # Apply update
            new_state[k] = global_state_cpu[k] + torch.from_numpy(weighted_update)

        model.load_state_dict(new_state)

        # ----------------------------------------
        # EVALUATION
        # ----------------------------------------
        acc = evaluate_model(model, test_loader)
        asr = evaluate_asr(model, zero_loader)
        
        history['acc'].append(acc)
        history['asr'].append(asr)

        # Track Average Reputations
        mal_indices = np.where(malicious_flags)[0]
        ben_indices = np.where([not m for m in malicious_flags])[0]
        
        avg_rep_mal = Gamma[mal_indices].mean() if len(mal_indices) > 0 else 0.0
        avg_rep_ben = Gamma[ben_indices].mean()
        
        history['rep_mal'].append(avg_rep_mal)
        history['rep_ben'].append(avg_rep_ben)

        # Convergence Check
        if acc >= Config.CONV_ACC and history['conv_round'] == Config.ROUNDS:
             history['conv_round'] = rnd + 1

        print(f"Round {rnd+1:02d} | Acc: {acc:.4f} | ASR: {asr:.4f} | Rep(Ben): {avg_rep_ben:.3f} | Rep(Mal): {avg_rep_mal:.3f}")

    return history


# ======================================================
# 6. PLOTTING
# ======================================================
def plot_results(all_results, output_dir="cifar10_v4_results"):
    if os.path.exists(output_dir) == False:
        os.makedirs(output_dir)

    # 1. Accuracy
    for dist in all_results:
        plt.figure(figsize=(10, 6))
        for m in all_results[dist]:
            plt.plot(all_results[dist][m]['acc'], label=f"Malicious={m}")
        plt.title(f"Accuracy vs Rounds ({dist}) - CKA Defense")
        plt.xlabel("Rounds")
        plt.ylabel("Accuracy")
        plt.grid(True)
        plt.legend()
        plt.savefig(f"{output_dir}/acc_{dist}.png")
        plt.close()

    # 2. ASR (Attack Success Rate)
    for dist in all_results:
        plt.figure(figsize=(10, 6))
        for m in all_results[dist]:
            plt.plot(all_results[dist][m]['asr'], label=f"Malicious={m}")
        plt.title(f"ASR vs Rounds ({dist}) - CKA Defense")
        plt.xlabel("Rounds")
        plt.ylabel("ASR (0->9)")
        plt.grid(True)
        plt.legend()
        plt.savefig(f"{output_dir}/asr_{dist}.png")
        plt.close()
    
    # 3. Reputation Evolution
    for dist in all_results:
        for m in all_results[dist]:
            if m == 0: continue # No malicious clients to compare
            plt.figure(figsize=(10, 6))
            data = all_results[dist][m]
            plt.plot(data['rep_ben'], label='Benign Avg Rep', color='green')
            plt.plot(data['rep_mal'], label='Malicious Avg Rep', color='red', linestyle='--')
            plt.title(f"Reputation Evolution ({dist}, Mal={m})")
            plt.xlabel("Rounds")
            plt.ylabel("Reputation Score")
            plt.grid(True)
            plt.legend()
            plt.savefig(f"{output_dir}/rep_{dist}_m{m}.png")
            plt.close()

    # 4. Convergence
    plt.figure(figsize=(10, 6))
    for dist in all_results:
        mal_counts = sorted(list(all_results[dist].keys()))
        conv_rounds = [all_results[dist][m]['conv_round'] for m in mal_counts]
        plt.plot(mal_counts, conv_rounds, marker='o', label=dist)
    plt.title("Convergence Speed")
    plt.xlabel("Number of Malicious Clients")
    plt.ylabel("Round Reached 85% Acc")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{output_dir}/convergence.png")
    plt.close()


# ======================================================
# 7. MAIN EXECUTION
# ======================================================
if __name__ == "__main__":
    # Required for multiprocessing with PyTorch/CUDA
    mp.set_start_method("spawn", force=True)

    print(f"Running on: {Config.DEVICE}")

    # 1. Prepare Data
    transform = transforms.ToTensor()
    # Download once in main process
    trainset = torchvision.datasets.CIFAR10('./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10('./data', train=False, download=True, transform=transform)

    # General Test Loader
    test_loader = DataLoader(testset, batch_size=256, shuffle=False)

    # Zero Loader (For ASR Calculation) - Only keep label 0
    zero_indices = [i for i, (img, label) in enumerate(testset) if label == 0]
    zero_loader = DataLoader(Subset(testset, zero_indices), batch_size=256, shuffle=False)

    # 2. Partition Data (IID vs Non-IID)
    data_indices = np.arange(len(trainset))
    targets = np.array(trainset.targets)
    
    # IID Partition
    iid_shuffled = np.random.permutation(data_indices)
    partitions_iid = np.array_split(iid_shuffled, Config.NUM_CLIENTS)
    partitions_iid = [list(p) for p in partitions_iid] # Convert to list

    # Non-IID Partition (Sort by label then shard)
    sorted_indices = np.argsort(targets)
    # Divide into 2*N shards, give 2 shards to each client
    num_shards = Config.NUM_CLIENTS * 2
    shards = np.array_split(sorted_indices, num_shards)
    shard_idxs = np.random.permutation(num_shards)
    
    partitions_noniid = [[] for _ in range(Config.NUM_CLIENTS)]
    for client_id in range(Config.NUM_CLIENTS):
        # Assign 2 random shards per client
        shard_1 = shards[shard_idxs[client_id * 2]]
        shard_2 = shards[shard_idxs[client_id * 2 + 1]]
        partitions_noniid[client_id] = np.concatenate([shard_1, shard_2]).tolist()

    all_experiments = {
        'IID': {'parts': partitions_iid, 'results': {}}, 
        'NonIID': {'parts': partitions_noniid, 'results': {}}
    }

    # 3. Run Experiments
    for dist_name, data_info in all_experiments.items():
        partitions = data_info['parts']
        
        for m_count in Config.MAL_COUNTS:
            # Randomly select malicious clients
            malicious_flags = [False] * Config.NUM_CLIENTS
            mal_indices = random.sample(range(Config.NUM_CLIENTS), m_count)
            for i in mal_indices:
                malicious_flags[i] = True
            
            print(f"\n========================================")
            print(f"Running {dist_name} | Malicious Clients: {m_count}")
            print(f"========================================")
            
            res = run_experiment(partitions, malicious_flags, test_loader, zero_loader)
            all_experiments[dist_name]['results'][m_count] = res

    # 4. Plot
    final_results = {k: v['results'] for k, v in all_experiments.items()}
    plot_results(final_results)
    print("\nDone! Plots saved to 'results_plots' directory.")