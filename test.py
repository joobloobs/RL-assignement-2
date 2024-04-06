import numpy as np
import gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from vars import *


# Define Policy Network
class Policy(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# REINFORCE Agent
class REINFORCEAgent:
    def __init__(self, input_dim, output_dim, hidden_dim=128, learning_rate=0.01, gamma=0.99):
        self.policy = Policy(input_dim, output_dim, hidden_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.gamma = gamma

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = torch.softmax(self.policy(state), dim=1)
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)

    def update_policy(self, rewards, log_probs):
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

ENVS = [HyperEnv('CartPole-v1', 1000, 1000, 128, 1e-4, 100, 0.99, 0.1, 128, 64), HyperEnv('Acrobot-v1', 200, 500, 256, 1e-3, 500, 0.99, 0.1, 128, 256)]

def mc_reinforce(hyper_env: HyperEnv):

    env = gym.make(hyper_env.name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = REINFORCEAgent(state_dim, action_dim, hidden_dim=hyper_env.layer_size1, gamma=hyper_env.gamma)

    num_episodes = 1000
    max_steps = 1000
    print_interval = 100
    scores = []

    for episode in range(num_episodes):
        state = env.reset()[0]
        rewards = []
        log_probs = []
        for step in range(max_steps):
            action, log_prob = agent.select_action(state)
            next_state, reward, done, _, _ = env.step(action)
            rewards.append(reward)
            log_probs.append(log_prob)
            if done:
                break
            state = next_state

        scores.append(sum(rewards))
        agent.update_policy(rewards, log_probs)

        if episode % print_interval == 0:
            mean_score = np.mean(scores[-print_interval:])
            print(f"Episode {episode}, Average Score: {mean_score}")

    env.close()
# Main
if __name__ == "__main__":
    mc_reinforce(ENVS[0])
