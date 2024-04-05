import torch
import torch.nn as nn
import torch.nn.functional as F


class DuelingQNetworkMean(nn.Module):
    """Implements a Dueling Q network (implements both value and advantage network with one shared layer)"""
    def __init__(self, n_states, n_actions, seed, hidden1_size=128, hidden2_size=128):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(DuelingQNetworkMean, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(n_states, hidden1_size)
        self.fc2 = nn.Linear(hidden1_size, hidden2_size)
        self.advfc3 = nn.Linear(hidden2_size, hidden2_size)
        self.advfc4 = nn.Linear(hidden2_size, n_actions)
        self.valfc3 = nn.Linear(hidden2_size, hidden2_size)
        self.valfc4 = nn.Linear(hidden2_size, 1)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        adv = F.relu(self.advfc3(x))
        adv = self.advfc4(adv)

        val = F.relu(self.valfc3(x))
        val = self.valfc4(val)

        return val + adv - adv.mean(1).unsqueeze(1)


class DuelingQNetworkMax(DuelingQNetworkMean):
    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        adv = F.relu(self.advfc3(x))
        adv = self.advfc4(adv)

        val = F.relu(self.valfc3(x))
        val = self.valfc4(val)

        return val + adv - adv.max(1)[0].unsqueeze(1)
