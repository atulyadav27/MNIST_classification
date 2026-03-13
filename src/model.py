import torch
torch.manual_seed(42)
from torch import nn
torch.manual_seed(42)
class MNIST_classifier(nn.Module):
  def __init__(self,input_shape,hidden_units,output_shape):
    super().__init__()
    self.layer_stack = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features = input_shape,out_features =hidden_units ),
        nn.ReLU(),
        nn.Linear(in_features = hidden_units, out_features = output_shape)

    )
  def forward(self,x):
    return self.layer_stack(x)
