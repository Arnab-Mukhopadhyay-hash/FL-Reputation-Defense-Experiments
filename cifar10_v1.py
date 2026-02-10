import os
import sys
import random
import time
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
import torch.multiprocessing as mp
from tqdm import tqdm

# -------------------- GLOBAL CONFIGURATION --------------------
# Hardware Settings
MAX_WORKERS = 2
OUTPUT_DIR = "cifar10_v1_results"

# Worker Training Config
LOCAL_EPOCHS = 1    # Increased to ensure local learning happens
BATCH_SIZE = 64
LR = 0.01             # Lowered slightly for stability

# FL Experiment Config
NUM_CLIENTS = 10
ROUNDS = 20
SEED = 1234
MALICIOUS_COUNTS = [0, 2, 4, 6]
MAL_TYPE = 'label_flip'

# Math Constants
EPS = 1e-10

# -------------------- MODEL DEFINITION --------------------
# Switched to a standard CNN for CIFAR-10. 
# ResNet18 is often too heavy/complex to train from scratch in FL on small data.
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

# -------------------- WORKER FUNCTION --------------------
def client_worker(args):
    """
    Independent worker function running in a separate process.
    """
    client_idx, global_weights, indices, malicious, mal_type, device_name, worker_seed, rnd = args
    
    # 1. Setup Environment
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    
    if torch.cuda.is_available() and "cuda" in device_name:
        device = torch.device(device_name)
    else:
        device = torch.device("cpu")
    
    # 2. Load Data Locally
    transform_local = transforms.Compose([
        transforms.RandomCrop(32, padding=4), 
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(), 
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Assumes data is already downloaded
    local_dataset = torchvision.datasets.CIFAR10(root="./data", train=True, download=False, transform=transform_local)
    
    # 3. Setup Model
    local_model = SimpleCNN().to(device)
    local_model.load_state_dict(global_weights)
    criterion = nn.CrossEntropyLoss()
    
    # Standard SGD is often more robust for FL than schedulers inside the worker
    opt = optim.SGD(local_model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)

    # 4. Pre-training Validation (Robust Subset)
    # Using a fixed subset (first 20%) ensures apples-to-apples comparison for loss
    split_point = int(len(indices) * 0.2)
    split_point = max(split_point, 50) # Ensure at least 50 validation samples
    val_indices = indices[:split_point]
    train_indices = indices[split_point:]

    # Validation Transform (No Augmentation)
    transform_eval = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    val_dataset = torchvision.datasets.CIFAR10(root="./data", train=True, download=False, transform=transform_eval)

    loader_val = DataLoader(Subset(val_dataset, val_indices), batch_size=BATCH_SIZE, shuffle=False)
    
    local_model.eval()
    loss_pre = 0.0; count = 0
    with torch.no_grad():
        for xb, yb in loader_val:
            xb = xb.to(device); yb = yb.to(device)
            out = local_model(xb)
            loss_pre += float(criterion(out, yb).item()) * xb.size(0)
            count += xb.size(0)
    loss_pre /= max(1, float(count))

    # 5. Prepare Training Data
    train_loader = DataLoader(Subset(local_dataset, train_indices), batch_size=BATCH_SIZE, shuffle=True)
    
    # Malicious Logic: Label Flipping
    if malicious and mal_type == 'label_flip':
        flip_from, flip_to = (0, 9)
        flipped_list = []
        # We define a temporary dataset wrapper or list to hold flipped labels
        # Note: We must iterate the subset to apply flips correctly
        temp_subset = Subset(local_dataset, train_indices)
        for i in range(len(temp_subset)):
            x, y = temp_subset[i]
            if int(y) == flip_from: y = flip_to
            flipped_list.append((x, y))
        train_loader = DataLoader(flipped_list, batch_size=BATCH_SIZE, shuffle=True)

    # 6. Training Loop
    local_model.train()
    for _ in range(LOCAL_EPOCHS):
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            out = local_model(xb)
            loss = criterion(out, yb)
            loss.backward()
            opt.step()

    # 7. Post-training Validation
    local_model.eval()
    loss_post = 0.0; count = 0
    with torch.no_grad():
        for xb, yb in loader_val:
            xb = xb.to(device); yb = yb.to(device)
            out = local_model(xb)
            loss_post += float(criterion(out, yb).item()) * xb.size(0)
            count += xb.size(0)
    loss_post /= max(1, float(count))

    # 8. Compute Gradients (Update) on CPU
    update = {}
    lstate = local_model.state_dict()
    for k, v in global_weights.items():
        local_v = lstate[k].cpu()
        upd = local_v - v 
        update[k] = upd.numpy().astype(np.float32)

    # Malicious Logic: Random Gradient
    if malicious and mal_type == 'random_grad':
        for k in update:
            update[k] = np.random.randn(*update[k].shape).astype(np.float32) * 0.05
            
    return update, float(loss_pre), float(loss_post)

# -------------------- HELPER FUNCTIONS --------------------
def partition_iid(dataset, num_clients):
    N = len(dataset)
    idxs = np.random.permutation(N)
    parts = np.array_split(idxs, num_clients)
    return [list(p) for p in parts]

def partition_noniid_labelskew(dataset, num_clients, shards_per_client=2):
    labels = np.array(dataset.targets)
    idxs = np.argsort(labels)
    shards = np.array_split(idxs, num_clients * shards_per_client)
    client_idxs = [[] for _ in range(num_clients)]
    shard_perm = np.random.permutation(len(shards))
    for i, s in enumerate(shard_perm):
        client_idxs[i % num_clients].extend(list(shards[s]))
    return client_idxs

def flatten_update(update):
    parts = []
    for k in sorted(update.keys()):
        parts.append(update[k].ravel())
    return np.concatenate(parts)

def update_norm(update):
    v = flatten_update(update)
    return np.linalg.norm(v)

def cosine_similarity_updates(a, b):
    A = flatten_update(a); B = flatten_update(b)
    den = (np.linalg.norm(A) * np.linalg.norm(B) + EPS)
    return float(np.dot(A, B) / den)

def pearson_correlation_updates(a, b):
    A = flatten_update(a)
    B = flatten_update(b)
    A_centered = A - A.mean()
    B_centered = B - B.mean()
    num = np.dot(A_centered, B_centered)
    den = (np.linalg.norm(A_centered) * np.linalg.norm(B_centered) + EPS)
    return float(num / den)

def get_opinion_from_metric(metric_val, W=2.0):
    val = min(max(metric_val, 0.0), 1.0)
    alpha = val
    beta = 1.0 - val
    denominator = alpha + beta + W
    return {'b': alpha/denominator, 'd': beta/denominator, 'u': W/denominator}

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
    # This was the Critical Fix: Add aggregated update to current weights
    # W_new = W_old + Sum(weight_i * update_i)
    global_dict = global_model.state_dict()
    new_state = {}
    
    # Calculate aggregated update first
    agg_update = {}
    first_k = list(updates[0].keys())
    
    for k in first_k:
        # stack updates for key k: shape (n_clients, *param_shape)
        # We compute linear combination manually to be safe
        agg_tensor = torch.zeros_like(updates[0][k])
        for i, u in enumerate(updates):
            agg_tensor += weights[i] * u[k]
        agg_update[k] = agg_tensor
        
    # Apply to model
    for k in global_dict:
        # W_new = W_old + Update
        new_state[k] = global_dict[k] + agg_update[k].to(global_dict[k].device)
        
    global_model.load_state_dict(new_state)


# -------------------- MAIN EXPERIMENT LOGIC --------------------
def experiment(partition, mal_count, device_list, test_data, loader_asr, mal_type='label_flip', tag='iid'):
    n_clients = len(partition)
    Gamma = np.ones(n_clients) / n_clients
    
    acc_log = []; asr_log = []
    reputations_log = []; rewards_log = []; fairness_log = []
    avg_rep_mal_log = []; avg_rep_ben_log = []
    convergence_round = None

    server_device = torch.device(device_list[0])
    global_model = SimpleCNN().to(server_device)

    # Parameters
    reputation_threshold_beta = 0.05
    nu = 0.5; delta = 0.85
    B = 1.0; B_w = 0.7 * B; B_r = B - B_w
    reward_exponent_q = 1.0
    lambda_penalty = 2.0; tau_penalty = 0.2
    rho_min = 0.9; pf_redistribute_fraction = 0.05
    W_param = 2.0 

    print(f"Starting {tag} (Mal: {mal_count})")

    malicious_flags = [False] * n_clients
    mal_idxs = np.random.choice(n_clients, mal_count, replace=False)
    for i in mal_idxs:
        malicious_flags[i] = True

    with mp.Pool(processes=MAX_WORKERS) as pool:
        for rnd in range(ROUNDS):
            global_state_cpu = {k: v.cpu() for k, v in global_model.state_dict().items()}
            
            client_tasks = []
            for i in range(n_clients):
                target_gpu = device_list[i % len(device_list)]
                worker_seed = SEED + rnd * 10000 + i
                task = (i, global_state_cpu, partition[i], malicious_flags[i], mal_type, target_gpu, worker_seed, rnd)
                client_tasks.append(task)

            # Parallel Execution
            results = []
            for res in tqdm(pool.imap(client_worker, client_tasks), total=n_clients, desc=f"Rd {rnd+1}", leave=False):
                results.append(res)
            
            updates = [r[0] for r in results]
            loss_pre_arr = [r[1] for r in results]
            loss_post_arr = [r[2] for r in results]

            # --- AGGREGATION LOGIC ---
            normed_updates = updates
            gamma_sum = np.sum(Gamma) + EPS
            agg_weights = Gamma / gamma_sum

            global_grad_ref = {}
            first_keys = list(normed_updates[0].keys())
            for k in first_keys:
                global_grad_ref[k] = np.zeros_like(normed_updates[0][k])
            for idx, u in enumerate(normed_updates):
                w = agg_weights[idx]
                for k in u.keys():
                    global_grad_ref[k] += w * u[k]

            s_list = []
            p_list = []
            for u in updates:
                s_list.append((cosine_similarity_updates(u, global_grad_ref) + 1.0)/2.0)
                p_list.append((pearson_correlation_updates(u, global_grad_ref) + 1.0)/2.0)

            delta_losses = [max(0.0, lp - lq) for lp, lq in zip(loss_pre_arr, loss_post_arr)]
            max_dl = max(delta_losses) if max(delta_losses) > 0 else 1.0
            q_list = [ dl / (max_dl + EPS) for dl in delta_losses ]

            b_list = []; u_list = [] 
            for s_i, p_i, q_i in zip(s_list, p_list, q_list):
                op_s = get_opinion_from_metric(s_i, W=W_param)
                op_p = get_opinion_from_metric(p_i, W=W_param)
                op_q = get_opinion_from_metric(q_i, W=W_param)
                
                op_sp = fuse_opinions(op_s, op_p)
                op_final = fuse_opinions(op_sp, op_q)
                
                total = op_final['b'] + op_final['d'] + op_final['u']
                b_list.append(op_final['b'] / total)
                u_list.append(op_final['u'] / total)

            Gamma_cand = np.array([ b + nu * u for (b,u) in zip(b_list, u_list) ], dtype=np.float64)
            Gamma = delta * Gamma + (1.0 - delta) * Gamma_cand
            Gamma = np.maximum(Gamma, 0.0)
            Gamma = Gamma / max(1.0, Gamma.max())

            mal_arr = np.array(malicious_flags, dtype=bool)
            avg_rep_mal_log.append(np.mean(Gamma[mal_arr]) if mal_arr.any() else 0.0)
            avg_rep_ben_log.append(np.mean(Gamma[~mal_arr]))

            R_idxs = [i for i in range(n_clients) if Gamma[i] >= reputation_threshold_beta]
            if len(R_idxs) == 0: R_idxs = list(range(n_clients))

            rep_for_agg = np.zeros(n_clients, dtype=np.float32)
            for i in R_idxs: rep_for_agg[i] = Gamma[i]
            srep = rep_for_agg.sum()
            rep_for_agg = rep_for_agg / srep if srep > 0 else np.ones(n_clients)/n_clients

            local_states = [{k: torch.tensor(v) for k, v in u.items()} for u in updates]
            aggregate_models_weighted(global_model, local_states, rep_for_agg)

            # Rewards
            pow_rep = np.power(Gamma + EPS, reward_exponent_q)
            if pow_rep.sum() == 0: pow_rep = np.ones_like(pow_rep)
            r_w = B_w * (pow_rep / pow_rep.sum())
            f_rep = np.square(Gamma + EPS)
            if f_rep.sum() == 0: f_rep = np.ones_like(f_rep)
            r_r = B_r * (f_rep / f_rep.sum())
            penalties = np.array([ lambda_penalty * max(0.0, tau_penalty - g)**2 for g in Gamma ])
            net_rewards = r_w + r_r - penalties

            # Fairness Redistribution
            v = Gamma.copy()
            def pearson_corr_simple(a, b):
                a = np.array(a); b = np.array(b)
                a = a - a.mean(); b = b - b.mean()
                denom = (np.sqrt((a*a).sum()) * np.sqrt((b*b).sum()) + EPS)
                return float((a*b).sum() / denom)
            
            rho = pearson_corr_simple(v, net_rewards)
            if rho < rho_min:
                fraction = pf_redistribute_fraction
                k = max(1, int(0.2 * n_clients))
                order = np.argsort(net_rewards)
                bottom = order[:k]; top = order[-k:]
                transfer = fraction * net_rewards[top].sum()
                if transfer > 0:
                    net_rewards[top] -= (net_rewards[top] / (net_rewards[top].sum()+EPS)) * transfer
                    net_rewards[bottom] += transfer / k

            # --- EVALUATION ---
            global_model.eval()
            
            # Accuracy
            correct = 0; total = 0
            with torch.no_grad():
                for xb, yb in DataLoader(test_data, batch_size=256):
                    xb = xb.to(server_device); yb = yb.to(server_device)
                    out = global_model(xb)
                    pred = out.argmax(dim=1)
                    correct += (pred == yb).sum().item()
                    total += yb.size(0)
            acc = correct / total
            acc_log.append(acc)

            # ASR
            asr_corr = 0; asr_total = 0
            with torch.no_grad():
                for xb, yb in loader_asr:
                    xb = xb.to(server_device); yb = yb.to(server_device)
                    out = global_model(xb)
                    pred = out.argmax(dim=1)
                    asr_corr += (pred == 9).sum().item() 
                    asr_total += yb.size(0)
            asr = asr_corr / max(1, asr_total)
            asr_log.append(asr)

            reputations_log.append(Gamma.copy())
            rewards_log.append(net_rewards.copy())
            fairness = (net_rewards[net_rewards>0].sum()**2) / (n_clients * (net_rewards**2).sum() + EPS)
            fairness_log.append(fairness)

            if convergence_round is None and acc >= 0.85:
                convergence_round = rnd + 1
            
            # PRINT EVERY ROUND as requested
            print(f"Round {rnd+1}/{ROUNDS} | Acc: {acc:.4f} | ASR: {asr:.4f} | Fair: {fairness:.3f}")

    return {
        'acc_log': np.array(acc_log),
        'asr_log': np.array(asr_log),
        'reputations_log': np.array(reputations_log),
        'avg_rep_mal_log': np.array(avg_rep_mal_log),
        'avg_rep_ben_log': np.array(avg_rep_ben_log),
        'rewards_log': np.array(rewards_log),
        'fairness_log': np.array(fairness_log),
        'convergence_round': convergence_round if convergence_round is not None else ROUNDS
    }

# -------------------- ENTRY POINT --------------------
if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    if os.path.exists(OUTPUT_DIR): shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if torch.cuda.is_available():
        DEVICE_LIST = [f'cuda:{i}' for i in range(torch.cuda.device_count())]
        print(f"✅ GPU Detected: {torch.cuda.get_device_name(0)}")
    else:
        DEVICE_LIST = ['cpu']
        print("⚠️ No GPU detected. Running on CPU.")
    
    print(f"🚀 Running with MAX_WORKERS = {MAX_WORKERS}")

    # Dataset Prep
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    _ = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    test_data = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    dataset_ref = torchvision.datasets.CIFAR10(root="./data", train=True, transform=transform)
    
    idx_zeros = [i for i, (x, y) in enumerate(test_data) if y == 0]
    loader_asr = DataLoader(Subset(test_data, idx_zeros), batch_size=256, shuffle=False)

    iid_part = partition_iid(dataset_ref, NUM_CLIENTS)
    noniid_part = partition_noniid_labelskew(dataset_ref, NUM_CLIENTS, shards_per_client=2)

    all_results = {'IID': {}, 'NonIID': {}}

    for tag, part in [('IID', iid_part), ('NonIID', noniid_part)]:
        for m in MALICIOUS_COUNTS:
            print(f"\n=== Running {tag} with {m} malicious clients ===")
            start_t = time.time()
            res = experiment(part, m, DEVICE_LIST, test_data, loader_asr, mal_type=MAL_TYPE, tag=tag)
            end_t = time.time()
            print(f"Finished in {end_t - start_t:.2f}s")
            
            all_results[tag][m] = res
            
            for k, v in res.items():
                if isinstance(v, (np.ndarray, list)):
                    np.save(os.path.join(OUTPUT_DIR, f"{tag}_mal{m}_{k}.npy"), v)

    # GENERATE PLOTS
    print("\nGenerating Plots...")

    # Accuracy
    plt.figure(figsize=(10,5))
    for m in MALICIOUS_COUNTS:
        plt.plot(all_results['IID'][m]['acc_log'], label=f"mal={m}")
    plt.title("Accuracy (IID)"); plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, "acc_iid.png")); plt.close()

    plt.figure(figsize=(10,5))
    for m in MALICIOUS_COUNTS:
        plt.plot(all_results['NonIID'][m]['acc_log'], label=f"mal={m}")
    plt.title("Accuracy (NonIID)"); plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, "acc_noniid.png")); plt.close()

    # ASR
    plt.figure(figsize=(10,5))
    for m in MALICIOUS_COUNTS:
        plt.plot(all_results['IID'][m]['asr_log'], label=f"mal={m}")
    plt.title("ASR (IID): 0 -> 9"); plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, "asr_iid.png")); plt.close()

    plt.figure(figsize=(10,5))
    for m in MALICIOUS_COUNTS:
        plt.plot(all_results['NonIID'][m]['asr_log'], label=f"mal={m}")
    plt.title("ASR (NonIID): 0 -> 9"); plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, "asr_noniid.png")); plt.close()

    # Final Accuracy
    plt.figure(figsize=(8,5))
    for tag in ['IID','NonIID']:
        finals = [all_results[tag][m]['acc_log'][-1] for m in MALICIOUS_COUNTS]
        plt.plot(MALICIOUS_COUNTS, finals, marker='o', label=tag)
    plt.title("Final Accuracy vs Malicious"); plt.grid(True); plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, "final_acc.png")); plt.close()

    # Convergence
    plt.figure(figsize=(8,5))
    for tag in ['IID','NonIID']:
        convs = [all_results[tag][m]['convergence_round'] for m in MALICIOUS_COUNTS]
        plt.plot(MALICIOUS_COUNTS, convs, marker='o', label=tag)
    plt.title("Convergence Rounds"); plt.grid(True); plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, "convergence.png")); plt.close()

    print("\nGenerating Reputation Evolution Plots for all scenarios...")

    tags = ['IID', 'NonIID']

    # Iterate through every combination of Tag and Malicious Count
    for tag in tags:
        for m in MALICIOUS_COUNTS:
            # 1. Retrieve the results for this specific setting
            # Note: Ensure 'all_results' follows the structure from your experiment loop
            if tag in all_results and m in all_results[tag]:
                res = all_results[tag][m]
                # 2. Identify correct keys (handles potential naming variations)
                key_mal = 'avg_rep_mal' if 'avg_rep_mal' in res else 'avg_rep_mal_log'
                key_ben = 'avg_rep_ben' if 'avg_rep_ben' in res else 'avg_rep_ben_log'
                plt.figure(figsize=(8, 5))
                # 3. Plot Benign Clients (Always present)
                plt.plot(res[key_ben], label='Benign Clients', linewidth=2, color='green')
                # 4. Plot Malicious Clients (Only if they exist)
                if m > 0:
                    plt.plot(res[key_mal], label='Malicious Clients', linewidth=2, linestyle='--', color='red')
                else:
                    # For 0 malicious, plot a flat zero line just for reference/legend
                    plt.plot(np.zeros_like(res[key_ben]), label='Malicious (None)', linewidth=2, linestyle=':', color='gray', alpha=0.5)

                # 5. Styling
                plt.title(f"Reputation Evolution ({tag}, Malicious={m})")
                plt.xlabel("Communication Round")
                plt.ylabel("Average Reputation Score (Gamma)")
                plt.legend(loc='best')
                plt.grid(True)
                plt.ylim(-0.05, 1.05) # Keep y-axis fixed [0,1] for easy comparison
                plt.tight_layout()
                
                # 6. Save
                filename = f"reputation_evolution_{tag}_mal{m}.png"
                save_path = os.path.join(OUTPUT_DIR, filename)
                plt.savefig(save_path)
                plt.close()
                
                print(f"Saved plot: {filename}")
            else:
                print(f"Skipping {tag} Mal={m} (Data not found)")

    print("Done generating all reputation plots.")

    print(f"Done! Results saved to '{OUTPUT_DIR}' directory.")