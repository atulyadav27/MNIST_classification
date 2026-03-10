import os

os.environ['HTTP_PROXY'] = "http://edcguest:edcguest@172.31.100.25:3128"
os.environ['HTTPS_PROXY'] = "http://edcguest:edcguest@172.31.100.25:3128"
import torch
import torchvision
from torchvision import datasets, transforms
transform = transforms.ToTensor()

train_dataset = datasets.MNIST(root = '../data',train = True, download = True,transform = transform)
test_dataset = datasets.MNIST(root = '../data',train = False, download = True,transform = transform)