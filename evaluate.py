import torch
from src.model import MNIST_classifier
from src.data import get_dataloaders
train_data_loaders, test_data_loaders = get_dataloaders()
from src.utils import accuracy_fn

model = MNIST_classifier(input_shape = 28*28,
                           hidden_units = 16,
                           output_shape = 10).to("cpu")
model.load_state_dict(torch.load("models/mnist_model.pth", map_location=torch.device("cpu")))
model.eval()
for batch , (X,Y) in enumerate(test_data_loaders):
    predict = model(X)
    print(predict.argmax(dim=1))
    print(Y)
    print(accuracy_fn(Y,predict.argmax(dim = 1)))
    break



