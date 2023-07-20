from abc import abstractmethod
import torch
import torch.nn as nn


class SpaceNetTemplate(nn.Module):
    """
    Template class for SpaceNet models.

    Todo
        * this could be an abstract class
    """

    def __init__(self, scale=0.25, lam = 1, device='cpu', **kwargs):
        super().__init__()
        self.scale = scale
        self.lam = lam
        self.device = device

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
        corr, p = self(x)
        label_corr = self.correlation_function(y)
        loss = torch.mean((corr - label_corr) ** 2)
        #loss = torch.mean(corr*label_corr)
        return loss + self.lam*torch.mean(p**2)

class OldSpaceNet(SpaceNetTemplate):
    """Feedforward SpaceNet model with a single hidden layer."""

    def __init__(self, n_in, n_out, **kwargs):
        super(OldSpaceNet, self).__init__(**kwargs)
        self.spatial_representation = torch.nn.Sequential(
            torch.nn.Linear(n_in, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, n_out),
            torch.nn.ReLU()
        )
        self.to(self.device)
        
    def forward(self, inputs):
        p = self.spatial_representation(inputs)  # ns, nr
        dp = torch.pdist(p)**2
        corr = torch.exp(-dp)
        return corr, p

    def correlation_function(self, r):        
        dr = torch.nn.functional.pdist(r)**2
        correlation = torch.exp(-0.5 / self.scale ** 2 * dr)
        return correlation

class ContextSpaceNet(OldSpaceNet):
    """An extension of the feedforward SpaceNet model that includes context.
    """
    
    def loss_fn(self, x, ys):
        """Loss function

        Args:
            inputs: Torch tensor of shape (batch size, 3). The function assumes
            that the first two components are spatial coordinates, while
            the last is a context coordinate.

        Returns:
            loss (1D tensor)
        """
        corr, p = self(x)
        
        labels = torch.ones_like(corr)

        for y in ys:
            labels *= self.correlation_function(y)

        loss = torch.mean((corr - labels) ** 2)
        return loss + self.lam*torch.mean(p**2)

class RecurrentSpaceNet(ContextSpaceNet):

    def __init__(
            self,
            n_in,
            n_out,
            initial_state_size=2,
            corr_across_space=False,
            num_layers=1,
            device='cpu',
            init_weights='identity',
            **kwargs
    ):
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
        num_layers: int
            Number of layers in the RNN.
        device: str
            The device this model should be used on.
        """
        super().__init__(n_in, n_out, **kwargs)
        self.corr_across_space = corr_across_space
        self.initial_state_size = initial_state_size
        self.num_layers = num_layers
        self.n_out = n_out
        self.device = device

        self.p0 = torch.nn.Sequential(
            torch.nn.Linear(initial_state_size, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, n_out * num_layers),
            torch.nn.ReLU()
        )

        self.spatial_representation = torch.nn.RNN(
            input_size=n_in,
            hidden_size=n_out,
            num_layers=num_layers,
            nonlinearity="relu",
            bias=False,
            batch_first=True
        )

        torch.nn.init.eye_(self.spatial_representation.weight_hh_l0)
        # self.init_weights(init_weights)
        self.to(device)

    def init_weights(self, method='identity'):
        """Initializes the weights of the model."""
        for i in range(self.num_layers - 1):
            if method == 'identity':
                torch.nn.init.eye_(getattr(self.spatial_representation, 'weight_hh_l' + str(i)))
            elif method == 'xavier':
                torch.nn.init.xavier_normal_(getattr(self.spatial_representation, 'weight_hh_l' + str(i)))
            elif method == 'normal':
                torch.nn.init.normal_(getattr(self.spatial_representation, 'weight_hh_l' + str(i)))
            elif method is None:
                pass
            else:
                raise ValueError('Unknown initialization method.')

    def correlation_function(self, r):
        """Computes the correlation function.

        Parameters
        ----------
        r: torch tensor
            Spatial coordinates of shape (batch size, time steps, 2).
        """
            
        if self.corr_across_space:
            # Flatten time-location dimension
            rr = torch.reshape(r, (-1, 2))  # bs*ts, 2
            dr = torch.nn.functional.pdist(rr)**2
        else:
            dr = torch.cdist(r, r)**2

        # Calculate correlation
        correlation = torch.exp(-0.5 / self.scale ** 2 * dr)
        return correlation  # save some computation

    def initial_state(self, initial_input):
        """Generates an initial state for the RNN.

        If initial_input is an integer, the input will just be static ones that are passed through the dense network.
        In that case, initial input is the batch size.

        If initial_input is a tensor, it will be passed through a linear layer directly. In this case, the current
        location, or a random tensor may be passed.
        """
        # Static initial state
        if isinstance(initial_input, int):
            initial_input = torch.ones(size=(initial_input, self.initial_state_size))

        # Make sure input is on correct device
        initial_input = initial_input.to(self.device)

        # Get initial state
        initial_state = self.p0(initial_input)

        # Reshape to (num_layers, batch size, n_out)
        initial_state = torch.reshape(initial_state, (self.num_layers, -1, self.n_out))

        return initial_state

    def forward(self, inputs):
        # Get initial state
        if isinstance(inputs, tuple):
            initial_state = self.initial_state(inputs[1])
            inputs = inputs[0]
        else:
            initial_state = self.initial_state(inputs.shape[0])

        # RNN returns representations and final hidden state
        p, _ = self.spatial_representation(inputs, initial_state)

        # Flatten across time and samples
        if self.corr_across_space:
            p = torch.reshape(p, (-1, p.shape[-1]))  # bsts, N
            dp = torch.nn.functional.pdist(p)**2
        else:
            dp = torch.cdist(p, p)**2

        corr = torch.exp(-dp) 
        
        return corr, p

class Decoder(torch.nn.Module):
    # Decodes from trained recurrent network states into Cartesian coordinates
    def __init__(self, n_in, n_out = 2, **kwargs):
        """ Dense network decoder

        Args:
            n_in (int): number of inputs features. 
            n_out (int): number of output features. Defaults to 2
        """
        
        super(Decoder, self).__init__(**kwargs)
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(n_in, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, n_out))
        self.mse = torch.nn.MSELoss()
        
    def forward(self, x):
        return self.decoder(x)

    def train_step(self, x, y, optimizer):
        optimizer.zero_grad()
        loss = self.loss_fn(x, y)
        loss.backward()
        optimizer.step()
        return loss.item()
       
    def loss_fn(self, x, y):
        return self.mse(self(x), y)
    
class End2End(RecurrentSpaceNet):
    def __init__(self, n_in, n_out, **kwargs):
        """ RNN trained end to end to decode into Cartesian coordinates

        Args:
            n_in (int): number of input features 
            n_out (int): number of output features. Defaults to 2.
        """
        super(End2End, self).__init__(n_in, n_out, **kwargs)
        self.decoder = Decoder(n_out, n_in)
        """ 
        torch.nn.Sequential(
            torch.nn.Linear(n_out, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, n_in))
        """
        self.mse = torch.nn.MSELoss()

    def forward(self, inputs):
        # RNN returns representations and final hidden state
        initial_state = self.p0(inputs[1])
        p, _ = self.spatial_representation(inputs[0], initial_state[None])
        return self.decoder(p)

    def loss_fn(self, x, y):
        return self.mse(self(x), y)

class StackedRecurrentSpaceNet(RecurrentSpaceNet):

    def __init__(
            self,
            n_in,
            n_out,
            initial_state_size=2,
            corr_across_space=False,
            num_layers=1,
            device='cpu',
            init_weights='identity',
            **kwargs
    ):
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
        num_layers: int
            Number of layers in the RNN.
        device: str
            The device this model should be used on.
        """

        self.rnn_layers = []

        # First layer
        self.rnn_layers.append(torch.nn.RNN(
            input_size=n_in,
            hidden_size=n_out,
            nonlinearity="relu",
            bias=False,
            batch_first=True
        ).to(device))

        # Additional layers
        for i in range(num_layers-1):
            self.rnn_layers.append(torch.nn.RNN(
                input_size=n_out,
                hidden_size=n_out,
                nonlinearity="relu",
                bias=False,
                batch_first=True
            ).to(device))

        self.init_weights(init_weights)

        super().__init__(n_in, n_out, initial_state_size, corr_across_space, num_layers, device, **kwargs)

    def init_weights(self, method='identity'):
        """Initializes the weights of the model."""
        for layer in self.rnn_layers:
            if method == 'identity':
                torch.nn.init.eye_(layer.weight_hh_l0)  # identity initialization
            elif method == 'xavier':
                torch.nn.init.xavier_normal_(layer.weight_hh_l0)
            elif method == 'normal':
                torch.nn.init.normal_(layer.weight_hh_l0)
            elif method is None:
                pass
            else:
                raise ValueError('Unknown initialization method.')

    def spatial_representation(self, inputs, initial_state):
        out = inputs
        for i, layer in enumerate(self.rnn_layers):
            out, hidden = layer(out, initial_state[i, None, :, :])
        return out, hidden

    def get_all_outputs(self, inputs, initial_state):
        """Returns lists of all outputs and hidden states of each of the rnn layers.
        """
        outputs, hiddens = [], []
        out = inputs
        for i, layer in enumerate(self.rnn_layers):
            out, hidden = layer(out, initial_state[i, None, :, :])
            outputs.append(out.detach())
            hiddens.append(hidden.detach())
        return outputs, hiddens

    def parameters(self, recurse: bool = True):
        parameters = list(self.p0.parameters())
        for layer in self.rnn_layers:
            parameters += list(layer.parameters())
        return parameters
