from abc import abstractmethod
import torch
import torch.nn as nn


class SpaceNetTemplate(nn.Module):
    """
    Template class for SpaceNet models.

    Todo
        * this could be an abstract class
    """

    def __init__(self, scale=0.4, **kwargs):
        super().__init__()
        self.scale = scale

    @abstractmethod
    def correlation_function(self, r):
        raise NotImplementedError

    def train_step(self, x, y, optimizer):
        """Perform a single training step and returns the loss.
        """
        optimizer.zero_grad()
        loss = self.loss_fn(x, y)
        loss.backward()
        optimizer.step()
        return loss.item()

    def loss_fn(self, x, y):
        # x = inputs to model, y = labels
        corr = self(x)
        label_corr = self.correlation_function(y)
        loss = torch.mean((corr - label_corr)**2)
        return loss


class OldSpaceNet(SpaceNetTemplate):
    """Feedforward SpaceNet model with a single hidden layer."""

    def __init__(self, n_in, n_out, scale=0.4, **kwargs):
        super(OldSpaceNet, self).__init__(scale, **kwargs)
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
        corr = p @ torch.transpose(p, dim0=-1, dim1=-2) # correlation matrix
        return corr

    def correlation_function(self, r):
        dr = torch.sum((r[:, None] - r[None])**2, dim = -1)
        correlation = torch.exp(-0.5/self.scale**2*dr)
        return correlation


class ContextSpaceNet(OldSpaceNet):
    """An extension of the feedforward SpaceNet model that includes context.
    """

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


class RecurrentSpaceNet(SpaceNetTemplate):

    def __init__(self, n_in, n_out, initial_state_size, corr_across_space=False, **kwargs):
        """Initializes a basic recurrent SpaceNet model.

        Parameters
        ----------
        n_in: int
            Number of input features.
        n_out: int
            Number of output features.
        initial_state_size: int
            Size of the initial state.
        corr_across_space: bool
            If True, the correlation function is computed across space and time. Otherwise, only time.
        """
        super().__init__(**kwargs)
        self.corr_across_space = corr_across_space
        self.initial_state_size = initial_state_size
        
        self.p0 = torch.nn.Sequential(
                torch.nn.Linear(initial_state_size, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, n_out),
                torch.nn.ReLU()
        )

        self.spatial_representation = torch.nn.RNN(
            input_size = n_in,
            hidden_size = n_out,
            nonlinearity= "relu",
            bias=False,
            batch_first=True)
        
        torch.nn.init.eye_(self.spatial_representation.weight_hh_l0) # identity initialization        

    def correlation_function(self, r):
        # Compare across time, not samples
        if not self.corr_across_space:
            dr = torch.sum((r[:, :, None] - r[:, None])**2, dim=-1)

        # Compare across time and samples
        else:
            rr = torch.reshape(r, (-1, 2))  # bs*ts, 2
            dr = torch.sum((rr[:, None] - rr[None]) ** 2, dim=-1)

        correlation = torch.exp(-0.5/self.scale**2*dr) 
        return torch.triu(correlation, diagonal = 0) # save some computation
    
    def initial_state(self, initial_input):
        # random initial state
        if isinstance(initial_input, int):
            initial_input = torch.ones(size=(initial_input, self.initial_state_size))
        # Get initial state
        initial_state = self.p0(initial_input)
        return initial_state
    
    def forward(self, inputs):
        # RNN returns representations and final hidden state
        if isinstance(inputs, tuple):
            initial_state = self.initial_state(inputs[1])
            inputs = inputs[0]
        else:
            initial_state = self.initial_state(inputs.shape[0])
        p, _ = self.spatial_representation(inputs, initial_state[None])

        # Flatten across time and samples
        if self.corr_across_space:
            p = torch.reshape(p, (-1, p.shape[-1]))  # bsts, N
        corr = p@torch.transpose(p, dim0=-1, dim1=-2) # correlation matrix
        return torch.triu(corr, diagonal=0)


class DoubleRecurrentSpaceNet(SpaceNetTemplate):
    """SpaceNet model with two recurrent layers to mimic EC-HPC network.
    """

    def __init__(self, n_in, n_out, initial_state_size, corr_across_space=False, **kwargs):
        super().__init__(**kwargs)

        self.corr_across_space = corr_across_space
        self.initial_state_size = initial_state_size

        self.p0 = torch.nn.Sequential(
            torch.nn.Linear(initial_state_size, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, n_out),
            torch.nn.ReLU()
        )

        # Spatial representation network layer, 2 stacked rnn layers
        self.spatial_representation = torch.nn.RNN(
            input_size=n_in,
            hidden_size=n_out,
            num_layers=2,
            nonlinearity='relu',
            bias=False,
            batch_first=True
        )

        # Todo is this needed?
        self.gp = torch.nn.Sequential(
            torch.nn.Linear(n_out, 64, bias=False),
            torch.nn.ReLU(),
        )

        # Initialize weights of recurrent layers
        torch.nn.init.eye_(self.spatial_representation_l1.weight_hh_l0)
        torch.nn.init.eye_(self.spatial_representation_l2.weight_hh_l0)

    def correlation_function(self, r):
        # Compare across time, not samples
        dr = torch.sum((r[:,:,None] - r[:,None])**2, dim = -1)
        correlation = torch.exp(-0.5/self.scale**2*dr)
        return torch.triu(correlation, diagonal = 0) # save some computation

class RecurrentDenseSpaceNet(SpaceNetTemplate):
    """SpaceNet model with one recurrent layer, and a feedforward network to mimic EC-HPC network.
    """

    def __init__(self, n_in, n_out, scale=0.4, **kwargs):
        super().__init__(scale, **kwargs)

        # Initialization network
        self.g0 = torch.nn.Sequential(
            torch.nn.Linear(2, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, n_in),
            torch.nn.ReLU()
        )

        # Spatial representation network rnn layer
        self.spatial_representation_rnn = torch.nn.RNN(
            input_size=n_in,
            hidden_size=64,
            nonlinearity='relu',
            bias=False,
            batch_first=True
        )

        # Spatial representation network dense layer
        self.spatial_representation_dense = torch.nn.Sequential(
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, n_out),
            torch.nn.ReLU()
        )

    def correlation_function(self, r):
        # Compare across time, not samples
        dr = torch.sum((r[:, :, None] - r[:, None])**2, dim=-1)
        correlation = torch.exp(-0.5 / self.scale**2*dr)
        return torch.triu(correlation, diagonal=0) # save some computation
