from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.depth.factory import DepthConfig, DepthFactory
from src.detection.factory import DetectionFactory, DetectorConfig
from src.localization.localizer import Localizer

LOGGER = logging.getLogger(__name__)


def configure_logging() -> None:
    """Configure logging for the localization test script."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def parse_args() -> argparse.Namespace:
    """Parse optional command-line overrides for the test script."""
    parser = argparse.ArgumentParser(description="Run Guardian Vision localization test")
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Optional image path override",
    )
    return parser.parse_args()


def _resolve_config_path() -> Path:
    candidates = [
        PROJECT_ROOT / "config" / "config.yaml",
        PROJECT_ROOT / "src" / "config" / "config.yaml",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Could not find config/config.yaml or src/config/config.yaml")


def _load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file) or {}
    return raw if isinstance(raw, dict) else {}


def _resolve_image_path(config: dict[str, Any]) -> Path:
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
    """Run YOLO, depth estimation, and pedestrian localization end-to-end."""
    configure_logging()
    args = parse_args()

    config = _load_config(_resolve_config_path())
    paths_cfg = config.get("paths", {}) if isinstance(config.get("paths", {}), dict) else {}
    depth_cfg = config.get("depth", {}) if isinstance(config.get("depth", {}), dict) else {}
    detector_cfg = config.get("detector", {}) if isinstance(config.get("detector", {}), dict) else {}

    image_path = Path(args.image).resolve() if args.image else _resolve_image_path(config)
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Failed to load image: {image_path}")

    detector = DetectionFactory.create(
        DetectorConfig(
            model=str(detector_cfg.get("model", "yolo")),
            model_path=str(detector_cfg.get("model_path", "yolo11n.pt")),
            confidence_threshold=float(config.get("confidence_threshold", 0.0)),
            device=str(config.get("device", "cpu")),
            output_images_dir=str(paths_cfg.get("output_images", "outputs/images")),
            output_videos_dir=str(paths_cfg.get("output_videos", "outputs/videos")),
        )
    )
    depth_estimator = DepthFactory.create(
        DepthConfig(
            model=str(depth_cfg.get("model", "depth_anything")),
            model_path=str(depth_cfg.get("model_path", "")),
            device=str(config.get("device", "cpu")),
            output_dir="outputs/depth",
        )
    )

    detections = detector.detect_image(str(image_path))
    depth_map = depth_estimator.estimate_depth(image)
    localizer = Localizer(output_dir="outputs/localization")
    localizations = localizer.localize(detections=detections, depth_map=depth_map)
    output_path = localizer.visualize(image=image, localizations=localizations)

    if not localizations:
        print("No pedestrians detected.")
        print(f"Saved localization visualization to: {output_path}")
        return

    for localization in localizations:
        print(f"Pedestrian {localization.id}")
        print(f"Ground Point: {localization.ground_point}")
        print(f"Relative Depth: {localization.relative_depth:.4f}")
        print(f"Confidence: {localization.confidence:.2f}")
        print()

    print(f"Saved localization visualization to: {output_path}")
    LOGGER.info("Localization test completed for %s", image_path)


if __name__ == "__main__":
    main()
