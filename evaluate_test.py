from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm

# Same settings as train.py
MODEL_NAME = "efficientnet_b0"
MODEL_PATH = Path("outputs/best_model.pth")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 224

# Folder
TEST_DIR = Path("dataset") / "test"
# Transform (same as val)
tfm = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),
                         (0.229, 0.224, 0.225)),
])

def main():
    test_ds = datasets.ImageFolder(TEST_DIR, transform=tfm)
    print("✅ Test classes:", test_ds.classes)  # should be ['ai', 'real']

    loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=0)

    model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=2)
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()

    # Confusion matrix counters
    # rows = true, cols = predicted
    # order: 0=ai, 1=real
    cm = [[0, 0],
          [0, 0]]

    wrong = []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)

            logits = model(imgs)
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            for i in range(len(labels)):
                t = int(labels[i].item())
                p = int(preds[i].item())
                cm[t][p] += 1

                if t != p:
                    # save a few wrong examples
                    idx_in_ds = len(wrong)
                    if idx_in_ds < 10:
                        path, _ = test_ds.samples[(len(wrong))]  # fallback, may not match exactly
                    # We'll store just class ids here (simple)
                    wrong.append((t, p, float(probs[i][p].item())))

    total = sum(sum(r) for r in cm)
    correct = cm[0][0] + cm[1][1]
    acc = correct / total if total > 0 else 0.0

    print("\n📌 Confusion Matrix (rows=true, cols=pred)")
    print("          pred_ai  pred_real")
    print(f"true_ai     {cm[0][0]:>5}    {cm[0][1]:>5}")
    print(f"true_real   {cm[1][0]:>5}    {cm[1][1]:>5}")

    print(f"\n✅ Test accuracy: {acc:.4f}  ({correct}/{total})")

    if cm[0][1] == 0 and cm[1][0] == 0:
        print("\n✅ No mistakes on test set.")
    else:
        print("\n⚠️ Some mistakes happened (showing up to 10):")
        for t, p, conf in wrong[:10]:
            tname = test_ds.classes[t]
            pname = test_ds.classes[p]
            print(f"true={tname}  pred={pname}  conf={conf:.4f}")

if __name__ == "__main__":
    main()
