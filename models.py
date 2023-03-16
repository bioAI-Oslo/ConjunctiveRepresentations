import torch
import torch.nn as nn


class SpaceNet(torch.nn.Module):
    """Simple SpaceNet model with one hidden layer.
    """

    def __init__(self, nr):
        super(SpaceNet, self).__init__()
        self.input_layer = nn.Linear(2, 512)
        self.hidden_layer = nn.Linear(512, 512)
        self.output_layer = nn.Linear(512, nr)
        self.activation = nn.ReLU()

    def forward(self, r):
        # Forward pass
        r = self.activation(self.input_layer(r))
        r = self.activation(self.hidden_layer(r))
        p = self.activation(self.output_layer(r))

        # Calculate correlation matrix
        corr = p @ p.T

        return p, corr

    def hidden_output(self, r):
        r = self.activation(self.input_layer(r))
        r = self.activation(self.hidden_layer(r))
        return r
