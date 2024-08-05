import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import OrderedDict

class FraudDetectionModel(nn.Module):
    def __init__(self, input_dim) -> None:
        """constructor the set up the architecture of the neural network"""
        super(FraudDetectionModel, self).__init__()

        # 4 linear layers, progressively reducing the dimensions
        # 1 ReLU activation function
        # 1 sigmoid function for the final output
        self.layer1 = nn.Linear(input_dim, 64)
        self.layer2 = nn.Linear(64, 32)
        self.layer3 = nn.Linear(32, 16)
        self.layer4 = nn.Linear(16, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """method to define how data flwos through the network
        applying linear transformations and activation functions"""
        
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))
        x = self.sigmoid(self.layer4(x))

        return x
    
# a couple of helper functions
def create_model(input_dim):
    """method to create an instance of the FraudDetectionModel"""
    return FraudDetectionModel(input_dim)

def get_model_parameters(model):
    """method to extract the model parameters as numpy arrays"""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

def set_model_parameters(model, parameters):
    """method to set the model's parameters from a list of numpy arrays"""
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict = True)