import torch
import torch.nn as nn
import torch.nn.functional as F

# actions
ACTIONS = 6


class DeepQNet(nn.Module):

    def __init__(self, num_states, seed):
        super(DeepQNet, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.input_layer = nn.Linear(num_states, 128)
        self.hidden_layer2 = nn.Linear(128,64)
        self.output_layer = nn.Linear(64, ACTIONS)

    def forward(self, x):
        x = F.relu(self.input_layer(x))
        x = F.relu(self.hidden_layer2(x))
        return self.output_layer(x)
