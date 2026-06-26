from __future__ import annotations

import argparse
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
    """Configure console logging for batch depth estimation."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for batch processing."""
    parser = argparse.ArgumentParser(description="Run depth estimation on a folder of images")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="datasets",
        help="Directory containing RGB images",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="src/config/config.yaml",
        help="Path to project YAML config",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Optional cap for number of images to process",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    """Load and parse a YAML file into a dictionary."""
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file) or {}
    return raw if isinstance(raw, dict) else {}


def collect_images(dataset_dir: Path) -> list[Path]:
    """Collect supported image files from the provided directory."""
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    allowed_suffixes = {".jpg", ".jpeg", ".png", ".bmp"}
    images = [
        path
        for path in sorted(dataset_dir.iterdir())
        if path.is_file() and path.suffix.lower() in allowed_suffixes
    ]

    if not images:
        raise RuntimeError(f"No supported images found in {dataset_dir}")

    return images


def main() -> None:
    """Run depth estimation for all images in dataset directory."""
    configure_logging()
    args = parse_args()

    config = load_yaml((PROJECT_ROOT / args.config).resolve())
    depth_cfg = config.get("depth", {}) if isinstance(config.get("depth", {}), dict) else {}

    estimator = DepthFactory.create(
        DepthConfig(
            model=str(depth_cfg.get("model", "depth_anything")),
            model_path=str(depth_cfg.get("model_path", "")),
            device=str(config.get("device", "cpu")),
            output_dir="outputs/depth",
        )
    )

    dataset_dir = (PROJECT_ROOT / args.dataset_dir).resolve()
    images = collect_images(dataset_dir)
    if args.max_images is not None:
        images = images[: args.max_images]

    LOGGER.info("Starting batch depth estimation for %d image(s)", len(images))

    for index, image_path in enumerate(images, start=1):
        image = cv2.imread(str(image_path))
        if image is None:
            LOGGER.warning("Skipping unreadable image: %s", image_path)
            continue

        start_time = time.perf_counter()
        depth_map = estimator.estimate_depth(image)
        elapsed = time.perf_counter() - start_time
        vis_path = estimator.visualize(depth_map)

        LOGGER.info(
            "[%d/%d] %s | shape=%s | min=%.6f | max=%.6f | time=%.4fs | output=%s",
            index,
            len(images),
            image_path.name,
            depth_map.shape,
            float(np.min(depth_map)),
            float(np.max(depth_map)),
            elapsed,
            vis_path,
        )

    LOGGER.info("Batch depth estimation finished.")


if __name__ == "__main__":
    main()
