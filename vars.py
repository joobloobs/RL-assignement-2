import torch
from collections import namedtuple

HyperEnv = namedtuple('HyperEnvs', ['name', 'n_episodes', 'max_t', 'batch_size', 'lr', 'update_every', 'gamma', 'tau', 'layer_size1', 'layer_size2'])

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
LR = 5e-4               # learning rate
UPDATE_EVERY = 500      # how often to update the network (When Q target is present)
TAU = 0.1

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")