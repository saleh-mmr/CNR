from torch import nn


class DQNNetwork(nn.Module):
    def __init__(self, num_actions, input_dim):
        super(DQNNetwork, self).__init__()

        # Fully Connected (FC) model
        self.FC = nn.Sequential(
            nn.Linear(input_dim, 64),
            # nn.LeakyReLU(negative_slope=0.01),    # Relu activation function helps learn non-linear patterns.
            nn.Tanh(),
            nn.Linear(64, 64),
            # nn.LeakyReLU(negative_slope=0.01),
            nn.Tanh(),
            nn.Linear(64, num_actions)        # [Q_left, Q_right]  → choose max action
        )

        # Weight initialization
        for layer in self.FC:
            if isinstance(layer, nn.Linear):
                nn.init.uniform_(layer.weight, -0.05, 0.05)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)


    def forward(self, x):
        """
        Forward pass through the Q-network.

        Parameters:
        ----------
        x : Tensor
            Input state(s) as a tensor [batch_size, input_dim]

        Returns:
        -------
        Q-values for each possible action [batch_size, num_actions]
        """
        Q = self.FC(x)
        return Q                                    # Q = [Q(action=left), Q(action=right)]
