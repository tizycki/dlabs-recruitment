import torch
from torch import nn


class ProfileClassifier(nn.Module):
    """
    Fully connected network mapping demographic data into psychological profile
    """
    def __init__(self, input_size, output_size):
        """
        Architecture initialization
        :param input_size: number of input features
        :param output_size: number of output features
        """
        super(ProfileClassifier, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, output_size)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(0.35)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass function
        :param x: input X tensor
        :return: tensor (1D) with psychological profile
        """
        output = self.fc1(x)
        output = self.relu(output)
        output = self.dropout1(output)
        
        output = self.fc2(output)
        output = self.relu(output)
        output = self.dropout1(output)
        
        output = self.fc3(output)
        output = self.relu(output)
        output = self.dropout1(output)
        
        output = self.fc4(output)

        return output
