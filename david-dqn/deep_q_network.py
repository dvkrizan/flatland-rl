import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch as T

class DeepQNetwork(nn.Module):
    # Our Network states states as input and outputs the q values of the actions
    def __init__(self, lr, state_size, n_actions, hid_size1, hid_size2, chkpt_dir, name):
        super(DeepQNetwork, self).__init__()
        self.checkpoint_dir = chkpt_dir
        self.name = name
        self.checkpoint_file = os.path.join(self.checkpoint_dir, self.name)

        self.fc1 = nn.Linear(state_size, hid_size1)
        self.fc1 = nn.Linear(hid_size1, hid_size2)
        self.fc3 = nn.Linear(hid_size2, n_actions)

        self.optimizer = optim.RMSprop(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        print(f'Model {self.name} using device {self.device}')
        self.to(self.device)


    def forward(self, state):
        layer1 = F.relu(self.fc1(state))
        layer2 = F.relu(self.fc2(layer1))
        actions = self.fc3(layer2)

        return actions


    def save_checkpoint(self):
        print(f'Saving checkpoint of neural network model {self.name}')
        T.save(self.state_dict(), self.checkpoint_file)


    def load_checkpoint(self):
        print(f'Loading checkpoint of neural network model {self.name}')
        self.load_state_dict(T.load(self.checkpoint_file))





