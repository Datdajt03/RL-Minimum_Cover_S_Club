"""
train.py — Huấn luyện và đánh giá đúng với PPO agent mới
"""
import torch
import numpy as np
from time import time
from experiment_config import ExperimentConfig


def train_agent(agent, env, optimizer, device):
    """
    Chạy training loop:
    - Mỗi epoch = 1 episode (reset env, chạy đến done)
    - PPO agent tự tích lũy buffer và update sau BATCH_SIZE bước
    """
    config = ExperimentConfig()
    metrics = {
        'solution_quality': [],
        'computation_time': [],
        'convergence_rate': [],
        'cover_sizes': [],
        'action_probs': [],
    }
    start = time()

    for epoch in range(config.max_epochs):
        state = env.reset()
        done  = False
        ep_reward = 0.0
        ep_action_probs = []

        while not done:
            # Thu thập xác suất chọn toán tử
            probs = agent.get_action_probs(state)
            ep_action_probs.append(probs)

            # Chọn toán tử
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)

            # Lưu transition (chỉ PPO agent thật mới xử lý)
            agent.store_transition(state, action, reward, done)

            ep_reward += reward
            state = next_state

        # PPO update sau mỗi episode
        agent.update()

        elapsed = time() - start
        cover_size = env.get_cover_size()
        metrics['solution_quality'].append(ep_reward)
        metrics['computation_time'].append(elapsed)
        metrics['convergence_rate'].append(ep_reward / elapsed if elapsed > 0 else 0)
        metrics['cover_sizes'].append(cover_size)

        mean_ep_probs = np.mean(ep_action_probs, axis=0).tolist() if ep_action_probs else [0.0]*3
        metrics['action_probs'].append(mean_ep_probs)

    return metrics


def evaluate_agent(agent, env, device):
    """Chạy 5 episode evaluation (không học), trả về metrics."""
    num_episodes = 5
    rewards, cover_sizes = [], []

    for _ in range(num_episodes):
        state = env.reset()
        done  = False
        ep_reward = 0.0

        while not done:
            action = agent.select_action(state)
            state, reward, done, _ = env.step(action)
            ep_reward += reward

        rewards.append(ep_reward)
        cover_sizes.append(env.get_cover_size())

    return {
        'average_reward':   float(np.mean(rewards)),
        'solution_quality': float(np.mean(rewards)),
        'cover_size':       float(np.mean(cover_sizes)),
        'cover_size_min':   float(np.min(cover_sizes)),
        'fully_covered':    env.is_fully_covered(),
    }