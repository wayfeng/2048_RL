import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64, fc3_units = 64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
            fc3_units (int): Number of nodes in third hidden layer

        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.fc1 = nn.Linear(state_size, fc1_units)
        self.bn1 = nn.BatchNorm1d(fc1_units)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.bn2 = nn.BatchNorm1d(fc2_units)
        self.act2 = nn.ReLU()
        self.fc3 = nn.Linear(fc2_units, fc3_units)
        self.bn3 = nn.BatchNorm1d(fc3_units)
        self.act3 = nn.ReLU()
        self.fc4 = nn.Linear(fc3_units, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""

        x = self.fc1(state)
        x = self.act1(x)
        x = self.bn1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.bn2(x)
        x = self.fc3(x)
        x = self.act3(x)
        return self.fc4(x)


class DQN(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, action_size):
        """Initialize parameters and build model.
        Params
        ======
            action_size (int): Dimension of each action

        """
        super(DQN, self).__init__()

        conv1_out = 64
        conv2_out = 256
        conv3_out = 2048
        self.conv1 = nn.Conv2d(1, conv1_out, 2, 1)
        self.bn1 = nn.BatchNorm2d(conv1_out)
        self.conv2 = nn.Conv2d(conv1_out, conv2_out, 2, 1)
        self.bn2 = nn.BatchNorm2d(conv2_out)
        self.conv3 = nn.Conv2d(conv2_out, conv3_out, 2, 1)
        self.bn3 = nn.BatchNorm2d(conv3_out)
        #def out_size(size, kernel_size=2, stride=1):
            #return (size - (kernel_size - 1) - 1) // stride + 1
        #w = out_size(out_size(out_size(state_size)))
        #fc_input_size = w * w * conv3_out
        #self.fc = nn.Linear(fc_input_size, action_size)
        self.fc = nn.Linear(conv3_out, action_size) # assume input is 4x4 here

    def forward(self, state):
        """Build a network that maps state -> action values."""

        x = F.relu(self.bn1(self.conv1(state)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.fc(x.view(x.size(0), -1))

