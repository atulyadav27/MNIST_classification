import torch

torch.manual_seed(42)
from torch import nn
from model import MNIST_classifier
import data
train_data_loader,test_data_loader = data.get_dataloaders()

Model_0 = MNIST_classifier(input_shape = 28*28,
                           hidden_units = 16,
                           output_shape = 10).to("cpu")

#Set accuracy function;
from utils import accuracy_fn


# define loss function and optimizer;
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params = Model_0.parameters(),lr = 0.1)

#Train the model
from tqdm.auto import tqdm

Epochs = 5;
train_accuracy = []
test_accuracy = []
train_Loss = []
test_Loss = []
epoch_count = []

for Epoch in tqdm(range(Epochs)):
  Loss = 0
  train_acc = 0
  Model_0.train()
  for batch , (X,Y) in enumerate(train_data_loader):
    #Do the forward pass
    predict = Model_0(X)
    #Calculate the loss
    loss = loss_function(predict,Y)
    Loss +=loss
    train_acc += accuracy_fn(y_true = Y,y_pred = predict.argmax(dim = 1))
    #Set optimizer zero grad
    optimizer.zero_grad()
    #Backpropagation
    loss.backward()
    #Set the gradient descent
    optimizer.step()
  print(f"Explored {Epoch+1} epochs")
  Loss /= len(train_data_loader)
  train_Loss.append(Loss.item())
  train_acc /= len(train_data_loader)
  train_accuracy.append(train_acc)
  epoch_count.append(Epoch)
  
  Model_0.eval()
  test_loss =0
  test_acc = 0
  
  with torch.inference_mode():
    for x,y in test_data_loader:
      test_pred = Model_0(x)
      test_loss += loss_function(test_pred,y)
      test_acc += accuracy_fn(y_true=y,y_pred=test_pred.argmax(dim = 1))
    test_loss /= len(test_data_loader)
    test_Loss.append(test_loss.item())
    test_acc /= len(test_data_loader)
    test_accuracy.append(test_acc)
  
  
  print(f"loss:{Loss:.4f},Train Accuracy :{train_acc:.4f},Test Loss:{test_loss:.4f},Test Accuracy:{test_acc:.4f}")

torch.save(Model_0.state_dict(), "../models/mnist_model.pth")
import matplotlib.pyplot as plt

plt.plot(epoch_count,train_Loss, c = "g", label = "Train Accuracy ")
plt.plot(epoch_count,test_Loss,c ='r', label ="Test Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()






