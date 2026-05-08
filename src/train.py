import torch
import torch.nn as nn
import torch.optim as optim
from data_loader import get_data_loaders
from model import get_model
import os

def train_model():
    # Detect Apple Silicon GPU (MPS)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Training on: {device}")

    # Load loaders and model
    data_dir = 'data/real_vs_fake' 
    train_loader, val_loader, _ = get_data_loaders(data_dir)
    model = get_model().to(device)

    # Loss and Optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

    # Simple 5-epoch loop
    for epoch in range(5):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        print(f"Epoch {epoch+1} Complete. Avg Loss: {running_loss/len(train_loader):.4f}")

    # Save the results
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), 'models/deepfake_detector.pth')
    print("Training finished! Model saved.")

if __name__ == "__main__":
    train_model()