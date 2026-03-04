import cv2
from spoof_detector import detect_spoof

IMAGE_PATH = r"C:\Users\vishw\Documents\ai_image_detector\dataset\test\ai\surprised-woman1.png"# Try also a real image path and compare.

img = cv2.imread(IMAGE_PATH)
result = detect_spoof(img)

print("Spoof result:", result["is_spoof"])
print("Spoof score :", result["spoof_score"])
print("Reasons     :", result["reasons"])
print("Signals     :", result["signals"])