from torch import nn


class DQNNetwork(nn.Module):
    def __init__(self, num_actions, input_dim):
        super(DQNNetwork, self).__init__()

        # Fully Connected (FC) model
        self.FC = nn.Sequential(
            nn.Linear(input_dim, 48),
            nn.LeakyReLU(negative_slope=0.01),    # Relu activation function helps learn non-linear patterns.

            nn.Linear(48, 48),
            nn.LeakyReLU(negative_slope=0.01),

            nn.Linear(48, num_actions)        # [Q_left, Q_right]  → choose max action
        )


    def forward(self, x):
        Q = self.FC(x)
        return Q                                    # Q = [Q(action=left), Q(action=right)]
