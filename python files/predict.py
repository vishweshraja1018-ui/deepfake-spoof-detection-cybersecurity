from pathlib import Path
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import timm

# Must match train.py
MODEL_NAME = "efficientnet_b0"
MODEL_PATH = Path("outputs/best_model.pth")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 224

# Must match ImageFolder class order printed:
# ✅ Classes found by ImageFolder: ['ai', 'real']
CLASS_NAMES = ["ai", "real"]

tfm = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),
                         (0.229, 0.224, 0.225)),
])

def predict(image_path: str):
    img = Image.open(image_path).convert("RGB")
    x = tfm(img).unsqueeze(0).to(DEVICE)

    model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=2)
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)[0].cpu()

    pred_idx = int(torch.argmax(probs).item())
    pred_label = CLASS_NAMES[pred_idx]
    conf = float(probs[pred_idx].item())

    print(f"\nImage: {image_path}")
    print(f"Prediction: {pred_label.upper()} | confidence: {conf:.4f}")
    print(f"Probabilities => ai: {probs[0]:.4f}, real: {probs[1]:.4f}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print('Usage: python predict.py "path_to_image.jpg"')
    else:
        predict(sys.argv[1])
