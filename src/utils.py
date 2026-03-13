import torch
def accuracy_fn(y_true, y_pred):
  acc = torch.eq(y_true,y_pred).sum().item()
  acc /=len(y_true)
  acc *= 100
  return acc