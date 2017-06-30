import torch
import torch.nn as nn
from torch.autograd import Variable

class BCELossReg(nn.Module):
    """
    implements binary corss entropy loss with additional regularization
    for ratio = 1, normal BCE Loss
    for ratio < 1, the BCE Loss is computed and given another parameter n,
        a weighted sum is returned: ratio * BCELoss + (1-ratio) * n * BCELoss
        where the ratio defines the weight applied to the BCELoss
    """

    def __init__(self, ratio=1, size_averaged=True):
        """
        :param ratio: int, 0<=r<=1, defines ratio of weighted sum
        :param size_averaged: average for number of training examples
        """
        assert type(r) == float, f'Expected float type for r, got {type(ratio)}'
        assert 0<= ratio <= 1, f'Expectted ratio between 0 and 1, got {ratio}'

        super(BCELossReg, self).__init__()

        self.size_averaged = size_averaged
        self.ratio = Variable(torch.Tensor([ratio]))
        self.ratio.requires_grad = False

    def forward(self, input, target, n):
        """
        :param input: according to nn.BCELoss()
        :param target: according to nn.BCELoss()
        :param n: int, multiplyer for (1-ratio) part of weighted sum
        """
        assert type(n) == int, f'Expected int type for n, got {type(n)}'
        n = Variable(torch.Tensor([n]))
        n.requires_grad = False

        # compute the BCE Loss
        result = torch.nn.BCELoss(size_average=self.size_averaged)(input, target)

        # apply weighted sum according to defined ratio
        result = self.ratio * result + (1-self.ratio) * result * n


        return result


"""
#Example
# init
model = nn.Sequential(
    nn.Linear(5,3),
    nn.ReLU(),
    nn.Linear(3,1),
    nn.Sigmoid()
)
r = 0.9
n = 3

loss_f = BCELossReg(r)
optim = torch.optim.Adam(model.parameters(), lr=0.001)
optim.zero_grad()

# model forward pass
data = Variable(torch.Tensor([1,2,3,4,5]).view(1,-1))
prediction = model(data)
target = Variable(torch.Tensor([1]))

# loss and backward
loss = loss_f(prediction, target, n)

loss.backward()
optim.step()

print("Loss", loss)
"""
