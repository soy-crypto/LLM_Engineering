import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

a = torch.randn(3)
b = torch.zeros(2, 4)
c = torch.tensor([1, 2, 3])

print(a)
print(b)
print(c)

print(a.shape)
print(b.shape)
print(c.shape)


x = torch.randn(2, 3)
y = torch.randn(2, 3)

print(x)
print(y)

print(x + y)
print(x * y)


a = torch.randn(2, 4)
b = torch.randn(4, 3)
c = a @ b
print(c.shape)

##################
x = torch.randn(2, 4, 8)
print(x.shape)

y = x.view(2, 2, 4, 4)
print(y.shape)

z = x.transpose(1, 2)
print(z.shape)

##################
x = torch.tensor(2.0, requires_grad = True)
y = x ** 2
y.backward()
print(x.grad)

##################
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)

    def forward(self, x):
        return self.linear(x)


model = SimpleModel()

x = torch.randn(4, 10)
y = model(x)
print(y)
print(y.shape)

###########################################