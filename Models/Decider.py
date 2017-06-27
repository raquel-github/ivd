import torch
import torch.autograd as autograd
import torch.nn as nn

class Decider(nn.Module):

    def __init__(self, hidden_encoder_dim):
        """
        Parameters
        hidden_encoder_dim      Dimensionaly of the hidden state of the encoder
        """

        super(Decider, self).__init__()

        self.decider_model = nn.Sequential(
            nn.Linear(hidden_encoder_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, hidden_encoder):
        print(hidden_encoder.size())
        return self.decider_model(hidden_encoder[0])
