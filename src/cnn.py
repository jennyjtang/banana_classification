from torchvision import models
from torchvision.models import ResNet18_Weights
import torch.nn as nn

def ResNet_GAP_RD():
    """Pretrained ResNet18 with dropout for regularization"""
    weights = ResNet18_Weights.IMAGENET1K_V1
    model = models.resnet18(weights=weights)

    # Freeze early layers (speeds up training, focuses on fine-tuning later layers)
    for param in model.parameters():
        param.requires_grad = False
    for param in model.layer4.parameters():  # Unfreeze last layer for fine-tuning
        param.requires_grad = True

    # Replace FC with dropout + new FC
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),  # Add dropout (50% probability)
        nn.Linear(num_features, 2)
    )
    return model

# Alias for backwards compatibility
BananaCNN = ResNet_GAP_RD