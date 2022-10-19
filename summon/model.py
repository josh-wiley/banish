from torch import Tensor
from torch.nn import Module, Embedding, Linear, BatchNorm1d, Tanh


class Model(Module):
    def __init__(self, rows: int, columns: int):
        super().__init__()

        hidden_size = rows * columns

        # input layer
        self.C = Embedding(rows, columns)
        self.linear1 = Linear(columns, hidden_size, bias=False)
        self.batchnorm1 = BatchNorm1d(hidden_size)
        self.activation1 = Tanh()

        # hidden layer
        self.linear2 = Linear(hidden_size, hidden_size)
        self.batchnorm2 = BatchNorm1d(hidden_size)
        self.activation2 = Tanh()

        # output layer
        self.linear3 = Linear(hidden_size, columns)
        self.batchnorm3 = BatchNorm1d(columns)

    def forward(self, x: Tensor) -> Tensor:
        # input layer
        x = self.C(x)
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
