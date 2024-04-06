import torch
import torch.nn as nn
import torch.nn.functional as F
from vars import SEED


class DuelingQNetwork(nn.Module):
    """Implements a module for the dueling DQN algorithm
     (implements both value and advantage network with one shared layer)"""
    def __init__(self, state_size, n_actions, hidden1_size=128, hidden2_size=128, type="mean"):
        """Initialize parameters and networks.
            :param state_size (int): Dimension of each state
            :param n_actions (int): Number of action
            :param hidden1_size (int): Number of nodes in first hidden layer
            :param hidden2_size (int): Number of nodes in second hidden layer
            :param type (str): Type of aggregation
        """
        self.type = type
        super(DuelingQNetwork, self).__init__()
        torch.manual_seed(SEED)

        # Shared layers for value and advantage networks
        self.fc1 = nn.Linear(state_size, hidden1_size)
        self.fc2 = nn.Linear(hidden1_size, hidden2_size)

        # Layers for advantage network
        self.advfc3 = nn.Linear(hidden2_size, hidden2_size)
        self.advfc4 = nn.Linear(hidden2_size, n_actions)

        # Layers for value network
        self.valfc3 = nn.Linear(hidden2_size, hidden2_size)
        self.valfc4 = nn.Linear(hidden2_size, 1)

    def forward(self, state):
        """maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        # Get advantage of each action
        adv = F.relu(self.advfc3(x))
        adv = self.advfc4(adv)

        # Get value of state
        val = F.relu(self.valfc3(x))
        val = self.valfc4(val)

        # Aggregate value and advantage based on the type of aggregation
        if self.type == "mean":
            return val + adv - adv.mean(1).unsqueeze(1)
        elif self.type == "max":
            return val + adv - adv.max(1)[0].unsqueeze(1)
        else:
            raise ValueError("Unrecognized aggregation type: {} is not a valid type. Should be 'mean' or 'max'".format(self.type))


class ReinforceNetwork(nn.Module):
    """Policy network used for monte carlo reinforce algorithm"""

    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super(ReinforceNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class TDNetwork(nn.Module):
    """Policy network used for the baseline of monte carlo Reinforce"""
    def __init__(self, state_size, n_actions, hidden1_size=128):
        """Initialize parameters and networks.
            :param state_size (int): Dimension of each state
            :param n_actions (int): Number of action
            :param hidden1_size (int): Number of nodes in first hidden layer
            :param hidden2_size (int): Number of nodes in second hidden layer
            :param type (str): Type of aggregation
        """
        super(TDNetwork, self).__init__()

        # Shared layers for value and advantage networks
        self.fc1 = nn.Linear(state_size, hidden1_size)
        self.fc2 = nn.Linear(hidden1_size, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x



