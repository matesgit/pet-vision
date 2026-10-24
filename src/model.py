"""Model definitions: a small CNN from scratch and pretrained backbones."""
import torch.nn as nn
from torchvision import models


class SimpleCNN(nn.Module):
    """Baseline trained from scratch: 4 conv blocks + global pooling + linear head."""

    def __init__(self, num_classes: int):
        super().__init__()
        chans = [3, 32, 64, 128, 256]
        layers = []
        for cin, cout in zip(chans[:-1], chans[1:]):
            layers += [
                nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ]
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(256, num_classes))

    def forward(self, x):
        x = self.pool(self.features(x)).flatten(1)
        return self.classifier(x)


def build_model(name: str, num_classes: int, pretrained: bool = True,
                freeze_backbone: bool = False) -> nn.Module:
    weights = "DEFAULT" if pretrained else None
    if name == "simple_cnn":
        model = SimpleCNN(num_classes)
        head_prefix = "classifier"
    elif name == "resnet18":
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        head_prefix = "fc"
    elif name == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=weights)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
        head_prefix = "classifier"
    else:
        raise ValueError(f"unknown model: {name}")

    if freeze_backbone:  # "linear probe": train only the new head
        for n, p in model.named_parameters():
            p.requires_grad = n.startswith(head_prefix)
    return model
