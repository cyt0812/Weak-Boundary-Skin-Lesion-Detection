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


class SkinLesionClassifier(nn.Module):

    def __init__(
        self,
        num_classes=7
    ):

        super().__init__()

        # ==========================================
        # Feature Extraction
        # ==========================================

        self.features = nn.Sequential(

            # 4 -> 32
            ConvBlock(
                in_channels=4,
                out_channels=32
            ),

            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ),

            # 32 -> 64
            ConvBlock(
                in_channels=32,
                out_channels=64
            ),

            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ),

            # 64 -> 128
            ConvBlock(
                in_channels=64,
                out_channels=128
            ),

            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ),

            # 128 -> 256
            ConvBlock(
                in_channels=128,
                out_channels=256
            ),

            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            )
        )

        # ==========================================
        # Global Average Pooling
        # ==========================================

        self.global_pool = nn.AdaptiveAvgPool2d(
            output_size=1
        )

        # ==========================================
        # Classification Head
        # ==========================================

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                256,
                128
            ),

            nn.ReLU(inplace=True),

            nn.Dropout(
                p=0.3
            ),

            nn.Linear(
                128,
                num_classes
            )
        )

    def forward(self, x):

        # Feature extraction
        features = self.features(x)

        # Global average pooling
        pooled = self.global_pool(features)

        # Classification
        logits = self.classifier(pooled)

        return logits


# ============================================================
# Model Test
# ============================================================

if __name__ == "__main__":

    print("========== Classification Model Test ==========")

    device = torch.device("cpu")

    # 创建模型
    model = SkinLesionClassifier(
        num_classes=7
    )

    model = model.to(device)

    # 模拟一个 Batch
    x = torch.randn(
        4,
        4,
        256,
        256
    ).to(device)

    # 前向传播
    output = model(x)

    print()
    print("Input shape:")
    print(x.shape)

    print()
    print("Output shape:")
    print(output.shape)

    print()
    print("Output:")
    print(output)

    print()
    print("Model parameters:")

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    print(total_params)

    print()
    print("==============================================")