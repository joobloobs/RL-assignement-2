from replayBuffer import ReplayBuffer
from qnetwork import DuelingQNetwork
from vars import *
import torch
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np


def dqn(agent, env, n_episodes=100, max_t=500):
    scores_over_episodes = []

    for i_episode in range(1, n_episodes+1):
        state = env.reset()[0]
        score = 0
        for t in range(max_t):
            # Act
            action = agent.act(state)
            next_state, reward, done, _, _ = env.step(action)

            # Learn
            agent.step(state, action, reward, next_state, done)

            state = next_state
            score += reward
            if done:
                break

        scores_over_episodes.append(score)

        print('\rEpisode {}\tScore: {:.2f}'.format(i_episode, score), end="")

        if i_episode % 100 == 0:
            print('\rEpisode {}\tScore: {:.2f}'.format(i_episode, score))

    return scores_over_episodes


class DuelingDQNAgent:
    def __init__(self, state_size, action_size, seed, type: str, hyper_env: HyperEnv):

        self.state_size = state_size
        self.action_size = action_size
        random.seed(seed)
        self.hyper_env = hyper_env

        # Initialize Q network and Q target network depending on the type of aggregation specified in hyper_env
        self.q_network = DuelingQNetwork(state_size, action_size, seed, hyper_env.layer_size1,
                                         hyper_env.layer_size2, type).to(device)

        self.target_network = DuelingQNetwork(state_size, action_size, seed, hyper_env.layer_size1,
                                              hyper_env.layer_size2, type).to(device)

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=LR)

        # replay buffer
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, self.hyper_env.batch_size, seed)

        # timestep used for updating Q target only every update_every defined in self.hyper_env.update_every
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):

        # Add the experience to the memory
        self.memory.add(state, action, reward, next_state, done)

        # update the q_network when the memory has enough data
        if len(self.memory) >= self.hyper_env.batch_size:
            experiences = self.memory.sample()
            self.update(experiences, self.hyper_env.gamma)

        # update target network every update_every steps
        self.t_step = (self.t_step + 1) % self.hyper_env.update_every
        if self.t_step == 0:

            self.target_network.load_state_dict(self.q_network.state_dict())

    def act(self, state):

        state = torch.from_numpy(state).float().unsqueeze(0).to(device)

        # get the next action from current state
        self.q_network.eval()
        with torch.no_grad():
            action_values = self.q_network(state)
        self.q_network.train()

        # Softmax policy action selection
        tau = self.hyper_env.tau
        action_values_table = action_values.cpu().data.numpy()[0]
        probabilities = np.exp((action_values_table - np.max(action_values_table)) / tau) / np.sum(
            np.exp((action_values_table - np.max(action_values_table)) / tau))

        return int(np.random.choice(len(action_values_table), p=probabilities))

    def update(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences

        # Choose best next action from q_network and evaluate from target network (maximization bias)
        actions_next = self.q_network(next_states).detach().argmax(1).unsqueeze(1)
        next_targets = self.target_network(next_states).gather(1, actions_next)

        # Compute the targets of states
        q_targets = rewards + (gamma * next_targets * (1 - dones))

        # Compute the current q values
        q_expected = self.q_network(states).gather(1, actions)

        loss = F.mse_loss(q_expected, q_targets)

        # Perform backward propagation to minimize loss
        self.optimizer.zero_grad()
        loss.backward()

        """for param in self.q_network.parameters():
            param.grad.data.clamp_(-1, 1)"""

        self.optimizer.step()



