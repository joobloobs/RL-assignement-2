import torch.autograd

from networks import ReinforceNetwork, TDNetwork
from vars import *
import random
import torch.optim as optim
from torch.distributions import Categorical
import torch.nn as nn
import numpy as np
from collections import namedtuple


def mc_reinforce(agent, env, num_episodes, max_steps):
    scores_over_episodes = []

    print_interval = 100
    scores = []

    for episode in range(num_episodes):
        state = env.reset()[0]
        rewards = []
        log_probs = []
        baseline_states = []
        baseline_targets = []
        for step in range(max_steps):
            action, log_prob = agent.select_action(state)
            next_state, reward, done, _, _ = env.step(action)
            rewards.append(reward)
            log_probs.append(log_prob)
            if agent.baseline:
                baseline_states.append(state)
                baseline_targets.append(
                    reward + agent.gamma * agent.policy(torch.tensor(next_state).float()).max().detach().item())
            if done:
                break
            state = next_state

        scores.append(sum(rewards))
        scores_over_episodes.append(sum(rewards))
        if agent.baseline:
            agent.update_policy(rewards, log_probs, baseline_states, baseline_targets)
        else:
            agent.update_policy(rewards, log_probs)

        if episode % print_interval == 0:
            mean_score = np.mean(scores[-print_interval:])
            print(f"Episode {episode}, Average Score: {mean_score}")
    return scores_over_episodes


class REINFORCEAgent:
    def __init__(self, state_size, n_actions, layer_size=128, learning_rate=0.01, gamma=0.99, baseline=False):
        self.policy = ReinforceNetwork(state_size, n_actions, layer_size)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.baseline = baseline
        if self.baseline:
            self.optimizer_baseline = optim.Adam(self.policy.parameters(), lr=learning_rate)
            self.network_baseline = TDNetwork(state_size, n_actions, layer_size)

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = torch.softmax(self.policy(state), dim=1)
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)

    def update_policy(self, rewards, log_probs, baseline_states=None, baseline_targets=None):
        discounted_rewards = []
        cumulative_reward = 0
        for r in reversed(rewards):
            cumulative_reward = r + self.gamma * cumulative_reward
            discounted_rewards.insert(0, cumulative_reward)
        discounted_rewards = torch.tensor(discounted_rewards)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)

        policy_loss = []
        for log_prob, reward in zip(log_probs, discounted_rewards):
            policy_loss.append(-log_prob * reward)
        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()

        if self.baseline:
            baseline_loss = nn.MSELoss()
            baseline_targets = torch.tensor(baseline_targets).float().unsqueeze(1)  # Ensure matching dimensions
            baseline_states = torch.tensor(np.array(baseline_states)).float()
            baseline_values = self.network_baseline(baseline_states)
            loss = baseline_loss(baseline_values, baseline_targets)
            self.optimizer_baseline.zero_grad()
            loss.backward()
            self.optimizer_baseline.step()

