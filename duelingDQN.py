from replayBuffer import ReplayBuffer
from qnetwork import QNetwork
from vars import *


class DuelingDQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)

    def build_model(self, type):
        pass

    def update_target_network(self):
        pass

    def select_action(self, state):
        pass

    def train(self, state, action, reward, next_state, done):
        pass

