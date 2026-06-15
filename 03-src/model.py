from pathlib import Path
from typing import Any

import torch
from timm import create_model


MODEL_REGISTRY = {
    "mobilenetv3": "mobilenetv3_large_100",
    "mobilenetv3_large_100": "mobilenetv3_large_100",
    "efficientnet_b0": "efficientnet_b0",
    "efficientnetb0": "efficientnet_b0",
    "densenet121": "densenet121",
}
DEFAULT_MODEL_NAME = "mobilenetv3"
NUM_CLASSES = 2
CLASS_NAMES = ["real", "fake"]


def normalize_model_name(model_name: str) -> str:
    key = model_name.strip().lower().replace("-", "_")
    if key not in MODEL_REGISTRY:
        supported = ", ".join(sorted({"mobilenetv3", "efficientnet_b0", "densenet121"}))
        raise ValueError(f"Unsupported model '{model_name}'. Supported models: {supported}")
    return key


def model_output_name(model_name: str) -> str:
    key = normalize_model_name(model_name)
    if key in {"mobilenetv3", "mobilenetv3_large_100"}:
        return "mobilenetv3"
    if key in {"efficientnet_b0", "efficientnetb0"}:
        return "efficientnet_b0"
    return "densenet121"


def build_model(
    model_name: str = DEFAULT_MODEL_NAME,
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True,
) -> torch.nn.Module:
    """Create the classifier backbone used by the project."""
    timm_name = MODEL_REGISTRY[normalize_model_name(model_name)]
    return create_model(timm_name, pretrained=pretrained, num_classes=num_classes)


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    **metadata: Any,
) -> None:
    payload: dict[str, Any] = {
        "state_dict": model.state_dict(),
        "metadata": metadata,
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_weights(model: torch.nn.Module, checkpoint_path: str | Path, device: str | torch.device) -> dict[str, Any]:
    """Load either a plain state_dict or this project's checkpoint format."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    metadata: dict[str, Any] = {}

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        metadata = checkpoint.get("metadata", {})
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    return metadata
