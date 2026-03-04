from pathlib import Path
import cv2
import numpy as np
import torch
import timm
from torchvision import transforms
from PIL import Image

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

MODEL_NAME = "efficientnet_b0"
MODEL_PATH = Path("outputs/best_model.pth")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 224

CLASS_NAMES = ["ai", "real"]

tfm = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),
                         (0.229, 0.224, 0.225)),
])

def main(img_path: str):

    # Load model
    model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=2)
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()

    # Choose last conv layer
    target_layers = [model.conv_head]

    # Load image
    pil = Image.open(img_path).convert("RGB")

    # Resize image to model size (224x224)
    pil_resized = pil.resize((IMG_SIZE, IMG_SIZE))
    rgb_img = np.array(pil_resized).astype(np.float32) / 255.0

    input_tensor = tfm(pil_resized).unsqueeze(0).to(DEVICE)

    # GradCAM
    cam = GradCAM(model=model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=input_tensor)[0]

    cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    # Save output
    out_dir = Path("gradcam_outputs")
    out_dir.mkdir(exist_ok=True)

    out_path = out_dir / (Path(img_path).stem + "_cam.jpg")
    cv2.imwrite(str(out_path), cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR))

    print("✅ Grad-CAM saved at:", out_path)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python gradcam_run.py path_to_image.jpg")
    else:
        main(sys.argv[1])
