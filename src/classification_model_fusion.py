import torch
import torch.nn as nn


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class SkinLesionFeatureFusionClassifier(nn.Module):

    def __init__(self, num_classes=7):
        super().__init__()

        # RGB image branch
        self.image_branch = nn.Sequential(
            ConvBlock(3, 16),
            nn.MaxPool2d(kernel_size=2, stride=2),

            ConvBlock(16, 32),
            nn.MaxPool2d(kernel_size=2, stride=2),

            ConvBlock(32, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),

            ConvBlock(64, 128)
        )

        # Coarse mask branch
        self.mask_branch = nn.Sequential(
            ConvBlock(1, 8),
            nn.MaxPool2d(kernel_size=2, stride=2),

            ConvBlock(8, 16),
            nn.MaxPool2d(kernel_size=2, stride=2),

            ConvBlock(16, 32),
            nn.MaxPool2d(kernel_size=2, stride=2),

            ConvBlock(32, 64)
        )

        # Global feature pooling
        self.image_pool = nn.AdaptiveAvgPool2d(1)
        self.mask_pool = nn.AdaptiveAvgPool2d(1)

        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Linear(128 + 64, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3)
        )

        # Classification head
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, image, mask):

        # Extract image features
        image_features = self.image_branch(image)
        image_features = self.image_pool(image_features)
        image_features = torch.flatten(image_features, 1)

        # Extract lesion-region features
        mask_features = self.mask_branch(mask)
        mask_features = self.mask_pool(mask_features)
        mask_features = torch.flatten(mask_features, 1)

        # Fuse image and mask features
        fused_features = torch.cat(
            [image_features, mask_features],
            dim=1
        )

        fused_features = self.fusion(fused_features)

        # Classification
        logits = self.classifier(fused_features)

        return logits