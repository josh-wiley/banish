from torch import Tensor
from torch.nn import Module, Embedding, Linear, BatchNorm1d, Tanh

# create one model per data type
# numeric, boolean, and categorical data types can be converted to numbers
# free text data types require a vocabulary and should stored in a transformer
# 1) inputs are row ID and column ID, single value output
# 2) inputs are row ID, output is all related column values

# TODO: use stddev and other statistics to determine hidden layer size


class Numeric(Module):
    def __init__(self, columns: int):
        super().__init__()

        hidden_size = 10

        # input layer
        self.linear1 = Linear(1, hidden_size, bias=False)
        self.batchnorm1 = BatchNorm1d(hidden_size)
        self.activation1 = Tanh()

        # hidden layer
        self.linear2 = Linear(hidden_size, hidden_size, bias=False)
        self.batchnorm2 = BatchNorm1d(hidden_size)
        self.activation2 = Tanh()

        # output layer
        self.linear3 = Linear(hidden_size, columns, bias=False)
        self.batchnorm3 = BatchNorm1d(columns)

    def forward(self, x: Tensor) -> Tensor:
        # input layer
        x = self.linear1(x)
        x = self.batchnorm1(x)
        x = self.activation1(x)

        # hidden layer
        x = self.linear2(x)
        x = self.batchnorm2(x)
        x = self.activation2(x)

        # output layer
        x = self.linear3(x)
        x = self.batchnorm3(x)

        return x
