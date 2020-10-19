import torch
from torch import nn


class ProfileClassifier(nn.Module):
    """
    
    """
    def __init__(self, input_size, output_size):
        """
        Architecture initialization
        :param out_class_num: number of output classes
        :param hidden_size: size of hidden layer
        """
        super(ProfileClassifier, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, output_size)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass function
        :param x: input X tensor
        :return: tensor with prediction logits
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