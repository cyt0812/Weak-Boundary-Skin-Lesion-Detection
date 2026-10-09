from PIL import Image
import numpy as np

image_path = r"D:\GraduationProject\Skin-Lesion-Detection\datasets\ISIC2018\segmentation\ISIC2018_Task1-2_Training_Input\ISIC_0000000.jpg"
mask_path = r"D:\GraduationProject\Skin-Lesion-Detection\outputs\prediction_test\ISIC_0000000_mask.png"
output_path = r"D:\GraduationProject\Skin-Lesion-Detection\outputs\prediction_test\ISIC_0000000_overlay.png"

image = Image.open(image_path).convert("RGB").resize((256, 256))
mask = Image.open(mask_path).convert("L")

image_array = np.array(image).astype(np.float32)
mask_array = np.array(mask) > 0

overlay = image_array.copy()
red = np.zeros_like(image_array)
red[:, :, 0] = 255

overlay[mask_array] = overlay[mask_array] * 0.5 + red[mask_array] * 0.5

overlay = np.clip(overlay, 0, 255).astype(np.uint8)

Image.fromarray(overlay).save(output_path)
print("Overlay saved to:", output_path)
