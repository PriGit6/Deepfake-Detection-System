import torch
from torchvision import transforms
from PIL import Image
from model import get_model
import sys
import os

def predict_image(image_path):
    # 1. Setup Device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # 2. Load the Model
    model = get_model().to(device)
    model.load_state_back = torch.load('models/deepfake_detector.pth', map_location=device)
    model.eval()

    # 3. Define the same transformations used during training
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 4. Load and process the image
    try:
        img = Image.open(image_path).convert('RGB')
        img_t = transform(img).unsqueeze(0).to(device)
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    # 5. Inference
    with torch.no_grad():
        output = model(img_t)
        probabilities = torch.exp(output) # Convert log-softmax back to 0-1 range
        prob_fake, prob_real = probabilities[0].tolist()

    # 6. Display Results
    print(f"\n--- Analysis for: {os.path.basename(image_path)} ---")
    if prob_real > prob_fake:
        print(f"RESULT: ✅ REAL (Confidence: {prob_real*100:.2f}%)")
    else:
        print(f"RESULT: ⚠️ FAKE/DEEPFAKE (Confidence: {prob_fake*100:.2f}%)")
    print(f"Raw Probabilities -> Real: {prob_real:.4f}, Fake: {prob_fake:.4f}\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 src/predict.py path/to/your/image.jpg")
    else:
        predict_image(sys.argv[1])