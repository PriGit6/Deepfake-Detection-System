import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

def get_data_loaders(data_dir, batch_size=32):
    # Transformation: Resize to 224x224 and normalize for ResNet
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Loading datasets from your folders
    train_set = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=transform)
    val_set = datasets.ImageFolder(os.path.join(data_dir, 'valid'), transform=transform)
    test_set = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader 