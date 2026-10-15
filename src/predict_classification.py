
import argparse
import os

import numpy as np
import torch
from PIL import Image

from classification_model import SkinLesionClassifier


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "classification",
    "classification_rgb_best_macro_f1.pth",
)

DEVICE = torch.device("cpu")


def load_model():
    """Load the saved classification model and its class labels."""
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(
            f"Model checkpoint not found: {MODEL_PATH}"
        )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True,
    )

    class_names = checkpoint["class_names"]
    input_size = tuple(checkpoint["input_size"])

    model = SkinLesionClassifier(
        num_classes=checkpoint["num_classes"]
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    return model, class_names, input_size, checkpoint


def preprocess_image(image_path, input_size):
    """Apply the same RGB resizing and scaling used during training."""
    if not os.path.isfile(image_path):
        raise FileNotFoundError(
            f"Image file not found: {image_path}"
        )

    with Image.open(image_path) as img:
        image = img.convert("RGB")
        image = image.resize(
            (input_size[1], input_size[0]),
            Image.Resampling.BILINEAR,
        )
        image = np.asarray(image, dtype=np.float32) / 255.0

    # HWC -> CHW -> BCHW
    image_tensor = torch.from_numpy(image)
    image_tensor = image_tensor.permute(2, 0, 1).contiguous()
    image_tensor = image_tensor.unsqueeze(0)

    return image_tensor.to(DEVICE)


def predict(image_path):
    model, class_names, input_size, checkpoint = load_model()
    image_tensor = preprocess_image(image_path, input_size)

    with torch.no_grad():
        logits = model(image_tensor)
        probabilities = torch.softmax(logits, dim=1)[0]

    predicted_index = int(torch.argmax(probabilities).item())

    print("\n=== Skin Lesion Classification (Research Prototype) ===")
    print("Image:", os.path.abspath(image_path))
    print("Device:", DEVICE)
    print("Model checkpoint:", MODEL_PATH)
    print("Checkpoint epoch:", checkpoint["epoch"])
    print("Input size:", f"{input_size[0]} x {input_size[1]}")
    print("Predicted class:", class_names[predicted_index])
    print(
        "Model score:",
        f"{probabilities[predicted_index].item() * 100:.2f}%"
    )

    print("\nScores for all classes:")
    for index, name in enumerate(class_names):
        score = probabilities[index].item() * 100
        print(f"  {name:6s}: {score:6.2f}%")

    print(
        "\nWARNING: Research output only. "
        "This model is not a medical diagnostic tool. "
        "Scores are model probabilities, not calibrated "
        "clinical confidence."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the trained skin lesion classifier."
    )
    parser.add_argument(
        "image",
        help="Path to a local skin lesion image.",
    )
    args = parser.parse_args()

    try:
        predict(args.image)
    except Exception as error:
        print(f"\nPrediction failed: {error}")
        raise
