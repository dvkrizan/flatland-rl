import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch as T

class DeepQNetwork(nn.Module):
    # Our Network states states as input and outputs the q values of the actions
    def __init__(self, lr, n_actions, name, input_dims, chkpt_dir):
        super(DeepQNetwork, self).__init__()
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)

        # input_dims[0] number of input frames, 32 outgoing filters, 8 kernel size
        self.conv1 = nn.Conv2d(input_dims[0], 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
        # Calculate input dims to fully connected layer with function below
        fc_input_dims = self.calculate_conv_output_dims(input_dims)
        
        self.fc1 = nn.Linear(fc_input_dims, 512)
        self.fc2 = nn.Linear(512, n_actions)

        self.optimizer = optim.RMSprop(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        print(self.device)
        self.to(self.device)


    def calculate_conv_output_dims(self, input_dims):
        # First, pass in a matrix of zeros with a batch size of 1 and the input
        # dimension of the frame stack (4, 84 , 84)
        state = T.zeros(1, *input_dims) # torch.Size([1, 4, 84, 84])
        dims = self.conv1(state)    # torch.Size([1, 32, 20, 20])
        dims = self.conv2(dims)     # torch.Size([1, 64, 9, 9])
        dims = self.conv3(dims)     # torch.Size([1, 64, 7, 7])
    
        return int(np.prod(dims.size()))  # 3136


    def forward(self, state):
        conv1 = F.relu(self.conv1(state))
        conv2 = F.relu(self.conv2(conv1))
        conv3 = F.relu(self.conv3(conv2))
        # conv3 shape is Batch Size x n_filters x H x W
        # Reshape using pytorch view (like numpy reshape function)
        # to batch size, (n_filters x H x W) or 32, 3136
        # The function argument -1 means to flattent he rest of the dimensions
        conv_state = conv3.view(conv3.size()[0], -1)
        flat1 = F.relu(self.fc1(conv_state))
        actions = self.fc2(flat1)

        return actions


    def save_checkpoint(self):
        print(f'...saving checkpoint...')
        T.save(self.state_dict(), self.checkpoint_file)


    def load_checkpoint(self):
        print('...loading checkpoint...')
        self.load_state_dict(T.load(self.checkpoint_file))





