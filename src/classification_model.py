import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class SkinLesionClassifier(nn.Module):
    """A standalone seven-class RGB image classifier. It does not use segmentation masks."""

    def __init__(self, num_classes=7):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3, 32),
            nn.MaxPool2d(2, 2),
            ConvBlock(32, 64),
            nn.MaxPool2d(2, 2),
            ConvBlock(64, 128),
            nn.MaxPool2d(2, 2),
            ConvBlock(128, 256),
            nn.MaxPool2d(2, 2),
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, image):
        features = self.features(image)
        pooled = self.global_pool(features)
        return self.classifier(pooled)


if __name__ == "__main__":
    model = SkinLesionClassifier(num_classes=7)
    sample = torch.randn(2, 3, 128, 128)
    output = model(sample)
    print("Input shape:", tuple(sample.shape))
    print("Output shape:", tuple(output.shape))
    print("Parameters:", sum(p.numel() for p in model.parameters()))
