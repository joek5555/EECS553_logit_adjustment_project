# class that represents the model for training a fully connected neural network
# must pass in the number of features in the tabular data

import torch
import math

class FullConnectNN(torch.nn.Module):

    def __init__(self, num_features):
        super().__init__()

        self.num_features = num_features

        self.fc1 = torch.nn.Linear(self.num_features, 132)
        self.fc2 = torch.nn.Linear(132, 132)
        self.fc3 = torch.nn.Linear(132, 6)

        self.init_weights()

    def init_weights(self):
        for fc_layer in [self.fc1, self.fc2, self.fc3]:
            fc_in = fc_layer.weight.size(1)
            torch.nn.init.normal_(fc_layer.weight, 0.0, 1 / math.sqrt(fc_in/2))
            torch.nn.init.constant_(fc_layer.bias, 0.0)

    def forward(self,x):
        z = torch.nn.functional.relu(self.fc1(x))
        z = torch.nn.functional.relu(self.fc2(z))
        z = torch.nn.functional.relu(self.fc3(z))

        return z