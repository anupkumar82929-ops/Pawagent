"""CNN backbone + classifier head for breed prediction."""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


def build_model(num_classes: int) -> nn.Module:
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
    m = models.efficientnet_b0(weights=weights)
    in_f = m.classifier[1].in_features
    m.classifier[1] = nn.Linear(in_f, num_classes)
    return m


def load_trained_model(path, device: torch.device) -> tuple[nn.Module, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    num_classes = int(ckpt["num_classes"])
    model = build_model(num_classes)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    meta = {
        "folder_ids": ckpt.get("folder_ids", []),
        "labels": ckpt.get("labels", []),
    }
    return model, meta
