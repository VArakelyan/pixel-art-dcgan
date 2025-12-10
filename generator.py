import torch
import torch.nn as nn

class PixelArtGenerator(nn.Module):
    """
    The exact Generator architecture from the training notebook.
    """
    def __init__(self, noise_dim=100, num_classes=5, class_embed_dim=50, channels=3, feature_maps=64):
        super().__init__()
        self.noise_dim = noise_dim
        self.class_embed_dim = class_embed_dim
        self.num_classes = num_classes

        # Class embedding
        self.label_embedding = nn.Embedding(num_classes, class_embed_dim)

        # Input: noise + class embedding
        input_dim = noise_dim + class_embed_dim

        # Initial projection
        self.project = nn.Sequential(
            nn.Linear(input_dim, feature_maps * 8 * 4 * 4),
            nn.ReLU(True)
        )

        # Upsampling blocks
        self.main = nn.Sequential(
            # 4x4 -> 8x8
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(feature_maps * 8, feature_maps * 4, 3, 1, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.ReLU(True),

            # 8x8 -> 16x16
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(feature_maps * 4, feature_maps * 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.ReLU(True),

            # 16x16 -> 32x32
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(feature_maps * 2, feature_maps, 3, 1, 1, bias=False),
            nn.BatchNorm2d(feature_maps),
            nn.ReLU(True),

            # Output layer
            nn.Conv2d(feature_maps, channels, 3, 1, 1),
            nn.Tanh()  # Output in [-1, 1]
        )

    def forward(self, noise, labels):
        # Embed labels
        label_embed = self.label_embedding(labels)

        # Concatenate noise and label embedding
        gen_input = torch.cat([noise, label_embed], dim=1)

        # Project and reshape
        x = self.project(gen_input)
        x = x.view(x.size(0), -1, 4, 4)

        # Generate image
        output = self.main(x)

        return output