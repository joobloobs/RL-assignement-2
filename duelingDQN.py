from replayBuffer import ReplayBuffer
from qnetwork import DuelingQNetwork
from vars import *
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np


def dqn(agent, n_episodes=10000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):

    scores_window = deque(maxlen=100)
    ''' last 100 scores for checking if the avg is more than 195 '''

    scores_over_episodes=[]

    eps = eps_start
    ''' initialize epsilon '''

    for i_episode in range(1, n_episodes+1):
        state = agent.env.reset()
        score = 0
        for t in range(max_t):
            action = agent.select_action(state, eps)
            next_state, reward, done, _ = agent.env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break

        scores_window.append(score)
        scores_over_episodes.append(score)

        eps = max(eps_end, eps_decay*eps)
        ''' decrease epsilon '''

        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")

        if i_episode % 100 == 0:
           print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window)>=195.0:
           print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
           break
    return scores_over_episodes


class DuelingDQNAgent:
    def __init__(self, env, eps, seed):
        self.env = env
        self.eps = eps

        self.replay_buffer = ReplayBuffer(env.action_space.n, BUFFER_SIZE, BATCH_SIZE, seed)

        self.dueling_q_network = DuelingQNetwork(env.observation_space.shape, env.action_space.n, seed)
        self.optimizer = optim.Adam(self.dueling_q_network.parameters(), lr=LR)

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        q_values = self.dueling_q_network.forward(state)

        if random.random() > self.eps:
            return np.argmax(q_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.env.action_space.n))

    def step(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)
        if len(self.replay_buffer) > BATCH_SIZE:
            loss = self.compute_loss(self.replay_buffer.sample())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def compute_loss(self, experiences):
        states, actions, rewards, next_states, dones = experiences

        q_value = self.dueling_q_network(states).gather(1, actions)
        q_next = self.dueling_q_network(next_states)
        best_q_next = q_next.max(1)[0]
        q_expected = rewards + GAMMA * best_q_next * (1-dones)

        loss = F.mse_loss(q_value, q_expected)

        return loss



