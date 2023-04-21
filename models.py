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
        corr = p @ torch.transpose(p, dim0= -1, dim1 = -2)
        return p, corr

    def hidden_output(self, r):
        r = self.activation(self.input_layer(r))
        r = self.activation(self.hidden_layer(r))
        return r
    
class OldSpaceNet(torch.nn.Module):
    def __init__(self, n_in, n_out, scale = 0.4, **kwargs):
        super(OldSpaceNet, self).__init__(**kwargs)
        self.scale = scale
        self.spatial_representation = torch.nn.Sequential(
            torch.nn.Linear(n_in, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, n_out),
            torch.nn.ReLU()
        )

    def forward(self, inputs):
        p = self.spatial_representation(inputs) # ns, nr
        corr = p@torch.transpose(p, dim0 = -1, dim1 = -2) # correlation matrix
        return corr

    def train_step(self, x, y, optimizer):
        optimizer.zero_grad()
        loss = self.loss_fn(x, y)
        loss.backward()
        optimizer.step()
        return loss.item()
        
    def correlation_function(self, r):
        dr = torch.sum((r[:,None] - r[None])**2, dim = -1)
        correlation = torch.exp(-0.5/self.scale**2*dr)
        return correlation

    def loss_fn(self, x, y):
        # x = inputs to model, y = labels
        corr = self(x)
        label_corr = self.correlation_function(y)
        loss = torch.mean((corr - label_corr)**2)
        return loss
    
class ContextSpaceNet(OldSpaceNet):
    def loss_fn(self, x, y):
        """Loss function 
        Args:
            inputs: Torch tensor of shape (batch size, 3). The function assumes
            that the first two components are spatial coordinates, while 
            the last is a context coordinate. 

        Returns:
            loss (1D tensor)
        """
        corr = self(x)
        label_space = self.correlation_function(y[:,:-1])
        label_context = self.correlation_function(y[:,-1,None])
        label_corr = label_space*label_context
        loss = torch.mean((corr - label_corr)**2)
        return loss

class RecurrentSpaceNet(OldSpaceNet):
    def __init__(self, n_in, n_out, scale = 0.4, **kwargs):
        super().__init__(n_in, n_out, scale)
        
        self.g0 = torch.nn.Sequential(
                torch.nn.Linear(2, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, n_out),
                torch.nn.ReLU())

        self.spatial_representation = torch.nn.RNN(
            input_size = n_in,
            hidden_size = n_out,
            nonlinearity= "relu",
            bias=False,
            batch_first=True)
        
        #self.gp = torch.nn.Sequential(
        #    torch.nn.Linear(n_out, 64, bias = False),
        #    torch.nn.ReLU(),
        #)
        
        
        torch.nn.init.eye_(self.spatial_representation.weight_hh_l0) # identity initialization        
    
    def correlation_function(self, r):
        # Compare across time, not samples
        dr = torch.sum((r[:,:,None] - r[:,None])**2, dim = -1)
        correlation = torch.exp(-0.5/self.scale**2*dr) 
        return torch.triu(correlation, diagonal = 0) # save some computation
    
    def initial_state(self, input_shape):
        # random initial state
        s0 = torch.rand(size = (input_shape, 2))
        initial_state = self.g0(s0)
        return initial_state
    
    def forward(self, inputs):
        # RNN returns representations and final hidden state
        initial_state = self.initial_state(inputs.shape[0])
        p, _ = self.spatial_representation(inputs, initial_state[None])
        corr = p@torch.transpose(p, dim0 = -1, dim1 = -2) # correlation matrix
        return torch.triu(corr, diagonal = 0)