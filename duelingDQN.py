from replayBuffer import ReplayBuffer
from qnetwork import DuelingQNetworkMean, DuelingQNetworkMax
from vars import *
import torch
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
from collections import deque


def dqn(agent, env, n_episodes=100, max_t=500):

    scores_window = deque(maxlen=100)
    ''' last 100 scores for checking if the avg is more than 195 '''

    scores_over_episodes=[]

    ''' initialize epsilon '''

    for i_episode in range(1, n_episodes+1):
        state = env.reset()[0]
        score = 0
        for t in range(max_t):
            action = agent.act(state)
            next_state, reward, done, _, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break

        scores_window.append(score)
        scores_over_episodes.append(score)

        ''' decrease epsilon '''

        print('\rEpisode {}\tAverage Score: {:.2f}\tNumber of steps: {}'.format(i_episode, np.mean(scores_window), t), end="")

        if i_episode % 100 == 0:
           print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window)>=195.0:
           print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
           break


    return scores_over_episodes


class DuelingDQNAgent:
    def __init__(self, state_size, action_size, seed, type: str, hyper_env: HyperEnv):

        self.state_size = state_size
        self.action_size = action_size
        random.seed(seed)
        self.hyper_env = hyper_env

        # Q targets and
        if type == "mean":
            self.q_network = DuelingQNetworkMean(state_size, action_size, seed, hyper_env.layer_size1, hyper_env.layer_size2).to(device)
            self.target_network = DuelingQNetworkMean(state_size, action_size, seed, hyper_env.layer_size1, hyper_env.layer_size2).to(device)
        elif type == "max":
            self.q_network = DuelingQNetworkMax(state_size, action_size, seed, hyper_env.layer_size1, hyper_env.layer_size2).to(device)
            self.target_network = DuelingQNetworkMax(state_size, action_size, seed, hyper_env.layer_size1, hyper_env.layer_size2).to(device)
        else:
            raise Exception('The type is wrong')
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=LR)

        ''' Replay memory '''
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, self.hyper_env.batch_size, seed)

        ''' Initialize time step (for updating every UPDATE_EVERY steps)           -Needed for Q Targets '''
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):

        ''' Save experience in replay memory '''
        self.memory.add(state, action, reward, next_state, done)

        ''' If enough samples are available in memory, get random subset and learn '''
        if len(self.memory) >= self.hyper_env.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences, self.hyper_env.gamma)

        """ +Q TARGETS PRESENT """
        ''' Updating the Network every 'UPDATE_EVERY' steps taken '''
        self.t_step = (self.t_step + 1) % self.hyper_env.update_every
        if self.t_step == 0:

            self.target_network.load_state_dict(self.q_network.state_dict())

    def act(self, state):

        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.q_network.eval()
        with torch.no_grad():
            action_values = self.q_network(state)
        self.q_network.train()

        tau = self.hyper_env.tau

        ''' Epsilon-greedy action selection (Already Present) '''
        action_values_table = action_values.cpu().data.numpy()[0]
        probabilities = np.exp((action_values_table - np.max(action_values_table)) / tau) / np.sum(
            np.exp((action_values_table - np.max(action_values_table)) / tau))
        return int(np.random.choice(len(action_values_table), p=probabilities))

    def learn(self, experiences, gamma):
        """ +E EXPERIENCE REPLAY PRESENT """
        states, actions, rewards, next_states, dones = experiences

        ''' Get max predicted Q values (for next states) from target model'''
        q_targets_next = self.q_network(next_states).detach().max(1)[0].unsqueeze(1)

        ''' Compute Q targets for current states '''
        q_targets = rewards + (gamma * q_targets_next * (1 - dones))

        ''' Get expected Q values from local model '''
        q_expected = self.q_network(states).gather(1, actions)

        ''' Compute loss '''
        loss = F.mse_loss(q_expected, q_targets)

        ''' Minimize the loss '''
        self.optimizer.zero_grad()
        loss.backward()

        ''' Gradiant Clipping '''
        """ +T TRUNCATION PRESENT """
        for param in self.q_network.parameters():
            param.grad.data.clamp_(-1, 1)

        self.optimizer.step()



