from abc import abstractmethod
import torch
import torch.nn as nn


class SpaceNetTemplate(nn.Module):
    """
    Template class for SpaceNet models.

    Todo
        * this could be an abstract class
    """

    def __init__(self, scale=0.4, device='cpu', **kwargs):
        super().__init__()
        self.scale = scale
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
        corr = self(x)
        label_corr = self.correlation_function(y)
        loss = torch.mean((corr - label_corr) ** 2)
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
        p = self.spatial_representation(inputs)  # ns, nr
        corr = p @ torch.transpose(p, dim0=-1, dim1=-2)  # correlation matrix
        return corr

    def correlation_function(self, r):
        dr = torch.sum((r[:, None] - r[None]) ** 2, dim=-1)
        correlation = torch.exp(-0.5 / self.scale ** 2 * dr)
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
        label_space = self.correlation_function(y[:, :-1])
        label_context = self.correlation_function(y[:, -1, None])
        label_corr = label_space * label_context
        loss = torch.mean((corr - label_corr) ** 2)
        return loss


class RecurrentSpaceNet(SpaceNetTemplate):

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
        super().__init__(**kwargs)
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
        # Compare across time, not samples
        if not self.corr_across_space:
            dr = torch.sum((r[:, :, None] - r[:, None]) ** 2, dim=-1)

        # Compare across time and samples
        else:

            # Flatten time-location dimension
            rr = torch.reshape(r, (-1, 2))  # bs*ts, 2

            # Compute pairwise distance for flattened array
            dr = torch.sum((rr[:, None] - rr[None]) ** 2, dim=-1)

        # Calculate correlation
        correlation = torch.exp(-0.5 / self.scale ** 2 * dr)
        return torch.triu(correlation, diagonal=0)  # save some computation

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

        # Compute correlation
        corr = p @ torch.transpose(p, dim0=-1, dim1=-2)
        return torch.triu(corr, diagonal=0)


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
