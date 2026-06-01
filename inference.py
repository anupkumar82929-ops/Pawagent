"""Image → top-k breed predictions using saved checkpoint."""
from __future__ import annotations

import io
import json
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from config import CLASS_NAMES_PATH, IMAGE_SIZE, MODEL_PATH
from modeling import load_trained_model


_transform = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

_model = None
_device = None
_labels: list[str] = []


def _ensure_loaded():
    global _model, _device, _labels
    if _model is not None:
        return
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"No model at {MODEL_PATH}. Run: python scripts/build_class_names.py && python train.py"
        )
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _model, meta = load_trained_model(MODEL_PATH, _device)
    _labels = list(meta.get("labels") or [])
    if not _labels and CLASS_NAMES_PATH.is_file():
        data = json.loads(CLASS_NAMES_PATH.read_text(encoding="utf-8"))
        order = meta.get("folder_ids") or [c["folder_id"] for c in data.get("classes", [])]
        by_f = {c["folder_id"]: c["label"] for c in data["classes"]}
        _labels = [by_f.get(int(fid), str(fid)) for fid in order]


def predict_bytes(data: bytes, top_k: int = 5) -> list[dict]:
    _ensure_loaded()
    img = Image.open(io.BytesIO(data)).convert("RGB")
    x = _transform(img).unsqueeze(0).to(_device)
    with torch.no_grad():
        logits = _model(x)
        probs = torch.softmax(logits, dim=1)[0]
    k = min(top_k, probs.numel())
    p, idx = torch.topk(probs, k)
    out = []
    for score, i in zip(p.tolist(), idx.tolist()):
        label = _labels[i] if i < len(_labels) else f"class_{i}"
        out.append({"rank": len(out) + 1, "label": label, "confidence": round(float(score), 4)})
    return out


def model_ready() -> bool:
    return MODEL_PATH.is_file()
