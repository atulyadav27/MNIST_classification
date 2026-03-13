import os

os.environ['HTTP_PROXY'] = "http://edcguest:edcguest@172.31.100.25:3128"
os.environ['HTTPS_PROXY'] = "http://edcguest:edcguest@172.31.100.25:3128"
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Transform: convert images to tensors
transform = transforms.ToTensor()

def get_datasets(data_dir="../data"):
    train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform
    )

    return train_dataset, test_dataset


def get_dataloaders(batch_size=16):
    train_dataset, test_dataset = get_datasets()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, test_loader