import torch.nn as nn
from torchvision import models

def get_model(num_classes=2):
    # Load pre-trained weights
    model = models.resnet18(weights='DEFAULT')
    
    # Freeze layers (so we only train the new 'head' we add)
    for param in model.parameters():
        param.requires_grad = False
    
    # Modify the final fully connected layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, num_classes),
        nn.LogSoftmax(dim=1)
    )
    
    return model