import os
import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from scipy.stats import pearsonr
import gudhi  

# ==========================================
# 1. CONFIGURATION
# ==========================================
class Config:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_CLIENTS = 10
    MALICIOUS_RATIO = 0.7 
    ROUNDS = 20 # Reduced for demonstration
    LOCAL_EPOCHS = 1
    BATCH_SIZE = 64
    LR = 0.01
    MOMENTUM = 0.9
    
    SL_DISCOUNT = 0.8       
    REDIST_PERCENT = 0.05   
    TOPO_K_WEIGHTS = 500    
    SUBSPACE_DIM = 10       
    OUTPUT_DIR = "cifar10_v5_results"

os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

# ==========================================
# 2. MODEL & DATA
# ==========================================
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

def get_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    # Non-IID Split
    idxs = np.argsort(trainset.targets)
    shards = np.array_split(idxs, Config.NUM_CLIENTS)
    client_loaders = [DataLoader(Subset(trainset, s), batch_size=Config.BATCH_SIZE, shuffle=True) for s in shards]
    
    anchor_loader = DataLoader(Subset(testset, range(100)), batch_size=100)
    return client_loaders, DataLoader(testset, batch_size=256), anchor_loader

# ==========================================
# 3. METRIC CALCULATIONS
# ==========================================

def compute_fisher_distance(update_vec, ref_vec, model, anchor_loader):
    model.eval()
    fisher_diag = torch.zeros_like(update_vec)
    criterion = nn.CrossEntropyLoss()
    
    for data, target in anchor_loader:
        data, target = data.to(Config.DEVICE), target.to(Config.DEVICE)
        model.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        
        flat_grads = torch.cat([p.grad.view(-1) for p in model.parameters() if p.grad is not None])
        fisher_diag += flat_grads ** 2
        break 
        
    fisher_diag = torch.clamp(fisher_diag, min=1e-5)
    diff = update_vec - ref_vec
    return torch.sqrt(torch.sum(fisher_diag * (diff ** 2))).item()

def compute_grassmann_distance(update_vec, ref_vec):
    dim = Config.SUBSPACE_DIM
    L = (len(update_vec) // dim) * dim
    mat_u = update_vec[:L].view(dim, -1)
    mat_ref = ref_vec[:L].view(dim, -1)
    
    try:
        u_base, _, _ = torch.svd_lowrank(mat_u, q=dim)
        ref_base, _, _ = torch.svd_lowrank(mat_ref, q=dim)
        _, s, _ = torch.svd(torch.mm(u_base.t(), ref_base))
        return torch.norm(torch.acos(torch.clamp(s, -1.0+1e-7, 1.0-1e-7))).item()
    except:
        return 1.57

def compute_homology_distance(update_vec, ref_vec):
    def get_pd(vec):
        k = min(Config.TOPO_K_WEIGHTS, len(vec))
        vals, _ = torch.topk(torch.abs(vec), k)
        st = gudhi.SimplexTree()
        for i, v in enumerate(vals.cpu().numpy()):
            st.insert([i], filtration=-v)
        st.compute_persistence()
        return st.persistence_intervals_in_dimension(0)

    try:
        return gudhi.bottleneck_distance(get_pd(update_vec), get_pd(ref_vec))
    except:
        return 1.0

# ==========================================
# 4. SUBJECTIVE LOGIC
# ==========================================
# ==========================================
# 4. UPDATED SUBJECTIVE LOGIC
# ==========================================
class SubjectiveOpinion:
    def __init__(self, b=0.0, d=0.0, u=1.0, a=0.5):
        self.b, self.d, self.u, self.a = b, d, u, a
        
    def fuse(self, other, check_disagreement=False, threshold=0.5):
        """
        Modified Consensus Operator.
        If check_disagreement is True, we evaluate if the two opinions 
        are too far apart to be trusted.
        """
        # Calculate disagreement (L1 distance between belief/disbelief pairs)
        if check_disagreement:
            disagreement = abs(self.b - other.b) + abs(self.d - other.d)
            if disagreement > threshold:
                # Set high uncertainty if metrics conflict
                return SubjectiveOpinion(b=0.1, d=0.1, u=0.8, a=self.a)

        denom = self.u + other.u - self.u * other.u
        if denom <= 0: return self
        
        new_b = (self.b * other.u + other.b * self.u) / denom
        new_d = (self.d * other.u + other.d * self.u) / denom
        new_u = (self.u * other.u) / denom
        return SubjectiveOpinion(new_b, new_d, new_u, self.a)
    
    def discount(self, factor):
        nb, nd = self.b * factor, self.d * factor
        return SubjectiveOpinion(nb, nd, 1.0 - (nb + nd), self.a)

    def expected_belief(self):
        return self.b + self.a * self.u

# ==========================================
# 5. MODIFIED EXPERIMENT LOOP (Logic Excerpt)
# ==========================================

# ... [Inside the round loop in run_fedtgs] ...

        

# ... [Rest of aggregation remains the same] ...

def metric_to_opinion(score, scale=1.0):
    prob_mal = 1.0 - np.exp(-scale * score)
    return SubjectiveOpinion((1-prob_mal)*0.9, prob_mal*0.9, 0.1)

# ==========================================
# 5. EXPERIMENT LOOP
# ==========================================
def run_fedtgs():
    loaders, test_loader, anchor_loader = get_data()
    global_model = SimpleCNN().to(Config.DEVICE)
    n_params = sum(p.numel() for p in global_model.parameters())
    global_anchor = torch.zeros(n_params).to(Config.DEVICE)
    
    mal_indices = random.sample(range(Config.NUM_CLIENTS), int(Config.NUM_CLIENTS * Config.MALICIOUS_RATIO))
    client_opinions = [SubjectiveOpinion() for _ in range(Config.NUM_CLIENTS)]
    history = {k: [] for k in ['acc', 'asr', 'rep_ben', 'rep_mal', 'rew_ben', 'rew_mal']}

    for rnd in range(Config.ROUNDS):
        local_updates = []
        
        for i in range(Config.NUM_CLIENTS):
            m_k = copy.deepcopy(global_model)
            optimizer = optim.SGD(m_k.parameters(), lr=Config.LR, momentum=Config.MOMENTUM)
            
            for x, y in loaders[i]:
                x, y = x.to(Config.DEVICE), y.to(Config.DEVICE)
                if i in mal_indices: y = torch.where(y == 0, torch.tensor(9).to(Config.DEVICE), y)
                
                optimizer.zero_grad()
                nn.CrossEntropyLoss()(m_k(x), y).backward()
                optimizer.step()
            
            vec = torch.cat([p.data.view(-1) for p in m_k.parameters()]) - \
                  torch.cat([p.data.view(-1) for p in global_model.parameters()])
            local_updates.append(vec)

        # Defense Logic
        ref = torch.stack(local_updates).mean(0) if rnd == 0 else global_anchor
        rewards = np.zeros(Config.NUM_CLIENTS)

        for i in range(Config.NUM_CLIENTS):
            s_f = compute_fisher_distance(local_updates[i], ref, global_model, anchor_loader)
            s_g = compute_grassmann_distance(local_updates[i], ref)
            s_t = compute_homology_distance(local_updates[i], ref)
            
            # 1. Map scores to base opinions
            op_f = metric_to_opinion(s_f, 0.5)
            op_g = metric_to_opinion(s_g, 2.0)
            op_t = metric_to_opinion(s_t, 5.0)
            
            # 2. Fuse with Disagreement Check (Threshold e.g., 0.4)
            # We fuse f and g, then fuse the result with t
            consensus = op_f.fuse(op_g, check_disagreement=True, threshold=0.4)
            consensus = consensus.fuse(op_t, check_disagreement=True, threshold=0.4)
            
            # 3. Update History
            client_opinions[i] = client_opinions[i].discount(Config.SL_DISCOUNT).fuse(consensus)
            
            # 4. New Weighting Strategy: w = b * (1 - u)
            # This ensures only high belief AND low uncertainty updates contribute
            rewards[i] = client_opinions[i].b * (1 - client_opinions[i].u)

        # Robin Hood Redistribution
        sorted_idx = np.argsort(rewards)
        tax = np.sum(rewards[sorted_idx[-2:]] * Config.REDIST_PERCENT)
        rewards[sorted_idx[-2:]] *= (1 - Config.REDIST_PERCENT)
        rewards[sorted_idx[:2]] += (tax / 2.0)

        # Aggregation
        weights = torch.tensor(rewards / (rewards.sum() + 1e-9)).float().to(Config.DEVICE)
        aggr = (torch.stack(local_updates) * weights.view(-1, 1)).sum(0)
        
        # Update Global Model
        ptr = 0
        for p in global_model.parameters():
            p.data += aggr[ptr:ptr+p.numel()].view(p.shape)
            ptr += p.numel()
        global_anchor = 0.9 * global_anchor + 0.1 * aggr.detach()

        # Eval
        global_model.eval()
        corr, total, asr_h, t0 = 0, 0, 0, 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(Config.DEVICE), y.to(Config.DEVICE)
                out = global_model(x).argmax(1)
                corr += (out == y).sum().item()
                total += y.size(0)
                asr_h += ((y == 0) & (out == 9)).sum().item()
                t0 += (y == 0).sum().item()
        
        history['acc'].append(corr/total)
        history['asr'].append(asr_h/max(1, t0))
        history['rep_ben'].append(np.mean([rewards[i] for i in range(Config.NUM_CLIENTS) if i not in mal_indices]))
        history['rep_mal'].append(np.mean([rewards[i] for i in mal_indices]))
        
        print(f"Round {rnd+1} | Acc: {history['acc'][-1]:.3f} | ASR: {history['asr'][-1]:.3f}")

    return history

# ==========================================
# 6. PLOTTING & EXECUTION
# ==========================================
def plot_results(history):
    rounds = range(1, len(history['acc']) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # Left Plot: Accuracy and Attack Success Rate
    plt.subplot(1, 2, 1)
    plt.plot(rounds, history['acc'], 'b-o', label='Main Accuracy')
    plt.plot(rounds, history['asr'], 'r--s', label='Backdoor ASR')
    plt.title('Model Performance under 70% Malicious Ratio')
    plt.xlabel('Federated Round')
    plt.ylabel('Rate')
    plt.legend()
    plt.grid(True)
    
    # Right Plot: Reputation Evolution
    plt.subplot(1, 2, 2)
    plt.plot(rounds, history['rep_ben'], 'g-', label='Avg Benign Reputation')
    plt.plot(rounds, history['rep_mal'], 'r-', label='Avg Malicious Reputation')
    plt.title('Trust (Expected Belief) over Time')
    plt.xlabel('Federated Round')
    plt.ylabel('Reputation Score')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show() # This will pop up the window
    plt.savefig(f"{Config.OUTPUT_DIR}/fedtgs_results.png")
    print(f"Plot saved to {Config.OUTPUT_DIR}/fedtgs_results.png")

if __name__ == "__main__":
    print(f"Starting FedTGS-RH on {Config.DEVICE}...")
    # This executes the training loop
    hist = run_fedtgs()
    
    # This generates the visual output
    plot_results(hist)