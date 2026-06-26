from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.depth.factory import DepthConfig, DepthFactory


LOGGER = logging.getLogger(__name__)


def configure_logging() -> None:
    """Configure script logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _resolve_config_path() -> Path:
    """Resolve configuration path across supported layouts."""
    candidates = [
        PROJECT_ROOT / "config" / "config.yaml",
        PROJECT_ROOT / "src" / "config" / "config.yaml",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Could not find config/config.yaml or src/config/config.yaml")


def _load_config(config_path: Path) -> dict[str, Any]:
    """Load YAML configuration as a dictionary."""
    with config_path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file) or {}
    return raw if isinstance(raw, dict) else {}


def _resolve_image_path(config: dict[str, Any]) -> Path:
    """Resolve input image path from config or fallback dataset sample."""
    paths_cfg = config.get("paths", {}) if isinstance(config.get("paths", {}), dict) else {}
    configured = paths_cfg.get("image", "")
    if configured:
        candidate = (PROJECT_ROOT / configured).resolve()
        if candidate.exists():
            return candidate

    dataset_images = sorted((PROJECT_ROOT / "datasets").glob("*.*"))
    for img_path in dataset_images:
        if img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
            return img_path

    raise FileNotFoundError("No image found. Set paths.image in config or add an image to datasets/")


def main() -> None:
    """Run depth estimation and save a colorized depth visualization."""
    configure_logging()

    config_path = _resolve_config_path()
    config = _load_config(config_path)

    depth_cfg = config.get("depth", {}) if isinstance(config.get("depth", {}), dict) else {}
    model_name = str(depth_cfg.get("model", "depth_anything"))
    model_path = str(depth_cfg.get("model_path", ""))
    device = str(config.get("device", "cpu"))

    estimator = DepthFactory.create(
        DepthConfig(
            model=model_name,
            model_path=model_path,
            device=device,
            output_dir="outputs/depth",
        )
    )

    image_path = _resolve_image_path(config)
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Failed to load image: {image_path}")

    start_time = time.perf_counter()
    depth_map = estimator.estimate_depth(image)
    inference_time = time.perf_counter() - start_time

    output_path = estimator.visualize(depth_map)

    print(f"Image Size: {image.shape[1]}x{image.shape[0]}")
    print(f"Inference Time: {inference_time:.4f} s")
    print(f"Depth Map Shape: {depth_map.shape}")
    print(f"Minimum Depth: {float(np.min(depth_map)):.6f}")
    print(f"Maximum Depth: {float(np.max(depth_map)):.6f}")
    print(f"Saved Depth Visualization: {output_path}")

    LOGGER.info("Depth estimation completed for %s", image_path)

if __name__ == "__main__":
    main()
