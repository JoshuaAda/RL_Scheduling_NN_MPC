from torch import nn
import torch
import torch.nn.functional as F
#### Neural Network controller
class NeuralNetwork(nn.Module):
    """
    three layers deep fully connected network
    """
    def __init__(self,states,output):
        super().__init__()
        self.fc1 = nn.Linear(states, states+5)
        self.fc2 = nn.Linear(states+5, states)
        self.fc3 = nn.Linear(states, output)


    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x=self.fc3(x)
        return x
