from pathlib import Path
from typing import Any

import torch
from timm import create_model


DEFAULT_MODEL_NAME = "mobilenetv3"
NUM_CLASSES = 2
CLASS_NAMES = ["real", "fake"]

MODEL_REGISTRY = {
    "mobilenetv3": {
        "timm_name": "mobilenetv3_large_100",
        "output_name": "mobilenetv3",
        "display_name": "MobileNetV3",
    },
    "mobilenetv3_large_100": {
        "timm_name": "mobilenetv3_large_100",
        "output_name": "mobilenetv3",
        "display_name": "MobileNetV3",
    },
    "efficientnet_b0": {
        "timm_name": "efficientnet_b0",
        "output_name": "efficientnet_b0",
        "display_name": "EfficientNetB0",
    },
    "efficientnetb0": {
        "timm_name": "efficientnet_b0",
        "output_name": "efficientnet_b0",
        "display_name": "EfficientNetB0",
    },
    "densenet121": {
        "timm_name": "densenet121",
        "output_name": "densenet121",
        "display_name": "DenseNet121",
    },
}


def normalize_model_name(model_name: str) -> str:
    key = model_name.strip().lower().replace("-", "_")
    if key not in MODEL_REGISTRY:
        supported = ", ".join(sorted({"mobilenetv3", "efficientnet_b0", "densenet121"}))
        raise ValueError(f"Unsupported model '{model_name}'. Supported models: {supported}")
    return key


def model_output_name(model_name: str) -> str:
    return MODEL_REGISTRY[normalize_model_name(model_name)]["output_name"]


def model_display_name(model_name: str) -> str:
    return MODEL_REGISTRY[normalize_model_name(model_name)]["display_name"]


def build_mobilenetv3(num_classes: int = NUM_CLASSES, pretrained: bool = True) -> torch.nn.Module:
    """Build MobileNetV3-Large exactly as used in the notebook."""
    return create_model(
        "mobilenetv3_large_100",
        pretrained=pretrained,
        num_classes=num_classes,
    )


def build_efficientnetb0(num_classes: int = NUM_CLASSES, pretrained: bool = True) -> torch.nn.Module:
    """Build EfficientNet-B0 exactly as used in the notebook."""
    return create_model(
        "efficientnet_b0",
        pretrained=pretrained,
        num_classes=num_classes,
    )


def build_densenet121(num_classes: int = NUM_CLASSES, pretrained: bool = True) -> torch.nn.Module:
    """Build DenseNet121 exactly as used in the notebook."""
    return create_model(
        "densenet121",
        pretrained=pretrained,
        num_classes=num_classes,
    )


def build_model(
    model_name: str = DEFAULT_MODEL_NAME,
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True,
) -> torch.nn.Module:
    """Create one of the three notebook-backed CNN classifiers."""
    output_name = model_output_name(model_name)
    if output_name == "mobilenetv3":
        return build_mobilenetv3(num_classes=num_classes, pretrained=pretrained)
    if output_name == "efficientnet_b0":
        return build_efficientnetb0(num_classes=num_classes, pretrained=pretrained)
    if output_name == "densenet121":
        return build_densenet121(num_classes=num_classes, pretrained=pretrained)
    raise ValueError(f"Unsupported model: {model_name}")


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
