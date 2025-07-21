import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.functional import log_softmax, softmax
import numpy as np
import matplotlib.pyplot as plt
import random

device = torch.device('cpu')  # 'cuda' si disponible

class StochasticChain:
    def __init__(self, N, p=0.4):
        self.N = N
        self.p = p
        self.log_p = np.log(p)
        self.log_q = np.log(1 - p)
        self.true_R = [self.reward(i) for i in range(N + 1)]
        self.true_Z = np.sum(self.true_R)
        self.true_prob = self.true_R / self.true_Z

    def reward(self, x):
        n = self.N
        sigma = n / 10.0
        return np.max([1e-5, np.exp(- (x - n / 4)**2 / sigma**2) + np.exp(- (x - 3 * n / 4)**2 / sigma**2)])

class ForwardPolicy(nn.Module):
    def __init__(self, N, hidden=64):  # Augmenté à 64
        super().__init__()
        self.embed = nn.Embedding(N + 1, hidden)
        self.mlp = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),  # Ajout d'une couche pour plus de profondeur
            nn.Linear(hidden, 2)  # 0: stop, 1: continue
        )

    def forward(self, s):
        s = torch.tensor([s]).to(device)
        e = self.embed(s)
        return self.mlp(e)

class BackwardPolicy(nn.Module):
    def __init__(self, N, hidden=64):  # Augmenté à 64
        super().__init__()
        self.embed = nn.Embedding(N + 1, hidden)
        self.mlp = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),  # Ajout d'une couche
            nn.Linear(hidden, 3)  # 0: stop, 1: continue_same, 2: continue_prev
        )

    def forward(self, s):
        s = torch.tensor([s]).to(device)
        e = self.embed(s)
        return self.mlp(e)

def sample_trajectory(env, policy):
    s = 0
    trajectory = []
    while True:
        if s == env.N:
            a = 'stop'
            s_next = s
            trajectory.append((s, a, s_next))
            terminal_x = s
            break
        logits = policy(s)
        probs = softmax(logits, dim=1)[0].cpu().detach().numpy()
        a_idx = np.random.choice(2, p=probs)
        a = 'stop' if a_idx == 0 else 'continue'
        if a == 'stop':
            s_next = s
            trajectory.append((s, a, s_next))
            terminal_x = s
            break
        else:
            if random.random() < env.p:
                s_next = s + 1
            else:
                s_next = s
        trajectory.append((s, a, s_next))
        s = s_next
    return trajectory, terminal_x

def compute_tb_loss(trajectory, terminal_x, policy, backward, logZ, env, stoch=True):
    left = logZ.clone()
    for s, a, s_next in trajectory:
        logits_f = policy(s)
        a_idx = 0 if a == 'stop' else 1
        log_pf = log_softmax(logits_f, dim=1)[0, a_idx]
        left += log_pf
        if stoch:
            if a == 'stop':
                logp = 0.0
            else:
                if s_next == s + 1:
                    logp = env.log_p
                elif s_next == s:
                    logp = env.log_q
                else:
                    raise ValueError("Invalid transition")
                left += torch.tensor(logp).to(device)
    right = torch.tensor(np.log(env.reward(terminal_x))).to(device)
    for s, a, s_next in trajectory:
        logits_b = backward(s_next)
        if s_next == 0:
            logits_b[0, 2] = -1e9
        log_pb_logits = log_softmax(logits_b, dim=1)
        if a == 'stop':
            type_idx = 0
        elif a == 'continue':
            if s == s_next:
                type_idx = 1
            else:
                type_idx = 2
        log_pb = log_pb_logits[0, type_idx]
        right += log_pb
    loss = (left - right) ** 2
    return loss

def train(env, stoch=True):
    policy = ForwardPolicy(env.N).to(device)
    backward = BackwardPolicy(env.N).to(device)
    logZ = nn.Parameter(torch.tensor(0.0).to(device))
    optimizer = optim.Adam(list(policy.parameters()) + list(backward.parameters()) + [logZ], lr=0.005)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.2)  # Reduce LR après 10000 steps
    for _ in range(2000):  # Augmenté à 2000
        loss = 0.0
        for _ in range(8):  # Batch à 8
            traj, x = sample_trajectory(env, policy)
            loss += compute_tb_loss(traj, x, policy, backward, logZ, env, stoch)
        loss /= 8
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
    return policy

def get_sampled_prob(policy, env, num_samples=10000):  # Plus de samples pour précision
    hist = np.zeros(env.N + 1)
    for _ in range(num_samples):
        _, x = sample_trajectory(env, policy)
        hist[x] += 1
    return hist / num_samples

Ns = [5, 10, 20, 30]
kl_standard = []
kl_stoch = []
for N in Ns:
    env = StochasticChain(N, p=0.4)
    print(f"Training for N={N}")
    policy_standard = train(env, stoch=False)
    sampled_standard = get_sampled_prob(policy_standard, env)
    kl_std = np.sum(env.true_prob * np.log(env.true_prob / (sampled_standard + 1e-10)))
    kl_standard.append(kl_std)
    print(f"KL standard: {kl_std}")
    policy_stoch = train(env, stoch=True)
    sampled_stoch = get_sampled_prob(policy_stoch, env)
    kl_s = np.sum(env.true_prob * np.log(env.true_prob / (sampled_stoch + 1e-10)))
    kl_stoch.append(kl_s)
    print(f"KL stochastic: {kl_s}")

# Plot KL divergence
plt.figure(figsize=(10, 5))
plt.plot(Ns, kl_standard, label='Standard GFlowNet')
plt.plot(Ns, kl_stoch, label='Stochastic GFlowNet')
plt.xlabel('Environment Size (N)')
plt.ylabel('KL Divergence')
plt.title('Performance Comparison: Standard vs Stochastic GFlowNet')
plt.legend()
plt.grid(True)
plt.show()

# Distributions pour N=30
env = StochasticChain(Ns[-1], p=0.4)
policy_standard = train(env, stoch=False)
sampled_standard = get_sampled_prob(policy_standard, env)
policy_stoch = train(env, stoch=True)
sampled_stoch = get_sampled_prob(policy_stoch, env)
x_range = np.arange(env.N + 1)
plt.figure(figsize=(10, 5))
plt.plot(x_range, env.true_prob, label='True Distribution', linewidth=2)
plt.plot(x_range, sampled_standard, label='Standard GFlowNet', linestyle='--')
plt.plot(x_range, sampled_stoch, label='Stochastic GFlowNet', linestyle='--')
plt.xlabel('State')
plt.ylabel('Probability')
plt.title(f'Distribution Comparison for N={Ns[-1]}')
plt.legend()
plt.grid(True)
plt.show()