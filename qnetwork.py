import torch
import torch.nn as nn
import torch.nn.functional as F


class DuelingQNetwork(nn.Module):
    """Implements a Dueling Q network (implements both value and advantage network with one shared layer)"""
    def __init__(self, n_states, n_actions, seed, hidden1_size=128, hidden2_size=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(DuelingQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.layer_1 = nn.Linear(n_states, hidden1_size)

        self.layer_val2 = nn.Linear(hidden1_size, hidden2_size)
        self.layer_val3 = nn.Linear(hidden2_size, 1)

        self.layer_adv2 = nn.Linear(hidden1_size, hidden2_size)
        self.layer_adv3 = nn.Linear(hidden2_size, n_actions)

    def forward(self, state):
        """Build a network that maps state -> q values calculated thanks to value and advantage."""
        x = F.relu(self.layer_1(state))

        state_val = F.relu(self.layer_val2(x))
        advantage = F.relu(self.layer_adv2(x))

        state_val = self.layer_val3(state_val)
        advantage = self.layer_adv3(advantage)

        q_values = state_val + advantage - advantage.mean()
        return q_values
