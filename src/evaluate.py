import torch
from data_loader import get_data_loaders
from model import get_model

def evaluate():
    # 1. Setup Device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Evaluating on: {device}")

    # 2. Load Data (Only the Test Set)
    data_dir = 'data/real_vs_fake'
    _, _, test_loader = get_data_loaders(data_dir)

    # 3. Load Model and Weights
    model = get_model().to(device)
    model.load_state_dict(torch.load('models/deepfake_detector.pth'))
    model.eval() # Set to evaluation mode

    # 4. Accuracy Calculation
    correct = 0
    total = 0
    
    print("Running test images... this may take a moment.")
    with torch.no_grad(): # Disable gradient calculation for speed
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'--- FINAL TEST RESULTS ---')
    print(f'Total Images Tested: {total}')
    print(f'Final Accuracy: {accuracy:.2f}%')

if __name__ == "__main__":
    evaluate()