"""
agent.py — Triển khai đúng theo paper EVO-RL (ban2)

4 điều kiện so sánh:
  1. AdaptiveOperatorAgent     — PPO controller chọn trong 3 toán tử
  2. NoReinforcementLearningAgent — xác suất toán tử cố định (1/3 đều nhau)
  3. FixedOperatorStrategyAgent  — luôn chọn toán tử 0 (crossover)
  4. RandomOperatorSelectionAgent — chọn ngẫu nhiên đều

State vector (5 chiều) theo paper:
  z_t = [f_min, f_avg, rho_feas, phi_bar, tau]

Action space (3 toán tử):
  0 = structural_crossover
  1 = feasibility_preserving_mutation
  2 = diversification_mutation

PPO theo paper:
  hidden_dim = 64, lr = 0.0001, clip_eps = 0.2, entropy_coef = 0.01
  discount = 0.99, batch_size = 32, grad_clip = 1.0
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

from experiment_config import ExperimentConfig

config = ExperimentConfig()

NUM_OPERATORS = 3   # {crossover, repair_mut, diversify_mut}
STATE_DIM     = 5   # [f_min, f_avg, rho_feas, phi_bar, tau]
HIDDEN_DIM    = config.hidden_dim
CLIP_EPS      = config.clip_eps
ENTROPY_COEF  = config.entropy_coef
GAMMA         = config.discount
LR            = config.learning_rate
BATCH_SIZE    = config.batch_size
GRAD_CLIP     = config.grad_clip


# ── Shared MLP backbone ──────────────────────────────────────────────────────

class PolicyValueNet(nn.Module):
    """MLP 2 tầng, đầu ra policy + value (actor-critic cho PPO)."""
    def __init__(self, state_dim=STATE_DIM, hidden_dim=HIDDEN_DIM, n_actions=NUM_OPERATORS):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden_dim, n_actions)   # logits
        self.value_head  = nn.Linear(hidden_dim, 1)           # V(s)

    def forward(self, x):
        h = self.shared(x)
        logits = self.policy_head(h)
        value  = self.value_head(h).squeeze(-1)
        return logits, value

    def get_action(self, state_tensor):
        """Trả về (action, log_prob, value) để dùng trong PPO rollout."""
        logits, value = self.forward(state_tensor)
        dist   = Categorical(logits=logits)
        action = dist.sample()
        return action.item(), dist.log_prob(action), value


# ── PPO update ───────────────────────────────────────────────────────────────

class PPOBuffer:
    """Lưu trữ một đoạn rollout để cập nhật PPO."""
    def __init__(self):
        self.states, self.actions, self.rewards = [], [], []
        self.log_probs, self.values, self.dones = [], [], []

    def store(self, state, action, reward, log_prob, value, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)

    def clear(self):
        self.__init__()

    def compute_returns(self, last_value=0.0, gamma=GAMMA):
        """GAE-style returns (simplified: discounted returns)."""
        returns = []
        R = last_value
        for r, done in zip(reversed(self.rewards), reversed(self.dones)):
            R = r + gamma * R * (1 - float(done))
            returns.insert(0, R)
        return torch.tensor(returns, dtype=torch.float32)


def ppo_update(net, optimizer, buffer: PPOBuffer, n_epochs=4):
    """Cập nhật PPO theo paper: clip ratio ε=0.2, entropy β=0.01."""
    returns    = buffer.compute_returns()
    states     = torch.tensor(np.array(buffer.states), dtype=torch.float32)
    actions    = torch.tensor(buffer.actions, dtype=torch.long)
    old_lps    = torch.stack(buffer.log_probs).detach()
    values_old = torch.stack(buffer.values).detach()

    advantages = returns - values_old
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    for _ in range(n_epochs):
        logits, values = net(states)
        dist    = Categorical(logits=logits)
        new_lps = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        ratio = torch.exp(new_lps - old_lps)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * advantages

        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss  = F.mse_loss(values, returns)
        loss = policy_loss + 0.5 * value_loss - ENTROPY_COEF * entropy

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), GRAD_CLIP)
        optimizer.step()


# ── Agent 1: AdaptiveOperatorAgent (PPO) ────────────────────────────────────

class AdaptiveOperatorAgent:
    """EVO-RL agent: dùng PPO để chọn toán tử theo state 5 chiều."""
    def __init__(self, state_dim=STATE_DIM, action_dim=NUM_OPERATORS):
        self.net = PolicyValueNet(state_dim, HIDDEN_DIM, action_dim)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=LR)
        self.buffer = PPOBuffer()
        self._last_log_prob = None
        self._last_value    = None

    def select_action(self, state):
        """state: np.array shape (5,) — state vector theo paper."""
        s = torch.tensor(state, dtype=torch.float32)
        action, lp, val = self.net.get_action(s)
        self._last_log_prob = lp
        self._last_value    = val
        return action

    def get_action_probs(self, state):
        s = torch.tensor(state, dtype=torch.float32)
        with torch.no_grad():
            logits, _ = self.net(s)
            probs = F.softmax(logits, dim=-1)
        return probs.cpu().numpy().tolist()

    def store_transition(self, state, action, reward, done):
        self.buffer.store(
            state, action, reward,
            self._last_log_prob, self._last_value, done)

    def update(self):
        if len(self.buffer.states) >= BATCH_SIZE:
            ppo_update(self.net, self.optimizer, self.buffer)
            self.buffer.clear()

    # Compat: policy_network cho train.py gốc
    @property
    def policy_network(self):
        return self.net


# ── Agent 2: NoReinforcementLearningAgent ────────────────────────────────────

class NoReinforcementLearningAgent:
    """Không có RL — xác suất toán tử cố định đều (1/3 mỗi loại)."""
    def __init__(self, state_dim=STATE_DIM, action_dim=NUM_OPERATORS):
        self.policy_network = PolicyValueNet(state_dim, HIDDEN_DIM, action_dim)
        # Freeze hoàn toàn — không học
        for p in self.policy_network.parameters():
            p.requires_grad_(False)

    def select_action(self, state):
        return int(torch.randint(0, NUM_OPERATORS, (1,)).item())

    def get_action_probs(self, state):
        return [1.0/NUM_OPERATORS] * NUM_OPERATORS

    def store_transition(self, *args): pass
    def update(self): pass


# ── Agent 3: FixedOperatorStrategyAgent ──────────────────────────────────────

class FixedOperatorStrategyAgent:
    """Lịch trình cố định: xoay vòng 0→1→2→0→... mỗi lần gọi."""
    def __init__(self, state_dim=STATE_DIM, action_dim=NUM_OPERATORS):
        self.policy_network = PolicyValueNet(state_dim, HIDDEN_DIM, action_dim)
        for p in self.policy_network.parameters():
            p.requires_grad_(False)
        self._step = 0

    def select_action(self, state):
        op = self._step % NUM_OPERATORS
        self._step += 1
        return op

    def get_action_probs(self, state):
        probs = [0.0] * NUM_OPERATORS
        probs[self._step % NUM_OPERATORS] = 1.0
        return probs

    def store_transition(self, *args): pass
    def update(self): pass


# ── Agent 4: RandomOperatorSelectionAgent ────────────────────────────────────

class RandomOperatorSelectionAgent:
    """Chọn toán tử hoàn toàn ngẫu nhiên đều."""
    def __init__(self, state_dim=STATE_DIM, action_dim=NUM_OPERATORS):
        self.policy_network = PolicyValueNet(state_dim, HIDDEN_DIM, action_dim)
        for p in self.policy_network.parameters():
            p.requires_grad_(False)

    def select_action(self, state):
        return int(np.random.randint(0, NUM_OPERATORS))

    def get_action_probs(self, state):
        return [1.0/NUM_OPERATORS] * NUM_OPERATORS

    def store_transition(self, *args): pass
    def update(self): pass