import torch
import torch.nn as nn
import torch.nn.functional as F

# amount of hidden nodes and actions
HIDDEN_NODES = 64
ACTIONS = 6


class DeepQNet(nn.Module):

    def __init__(self, num_states, seed):
        super(DeepQNet, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(num_states, HIDDEN_NODES)
        self.fc2 = nn.Linear(HIDDEN_NODES, HIDDEN_NODES)
        self.fc3 = nn.Linear(HIDDEN_NODES, ACTIONS)

        # TODO: self ? or seed

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
