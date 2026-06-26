from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

import cv2

from src.config.loader import load_config
from src.detection.base import Detection
from src.detection.factory import DetectionFactory, DetectorConfig


def configure_logging() -> None:
    """Configure project-wide logging format and level."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for Guardian Vision entrypoint."""
    parser = argparse.ArgumentParser(description="Guardian Vision - Modular Research Framework")
    parser.add_argument(
        "--config",
        type=str,
        default="src/config/config.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Optional image path override",
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Optional video path for person detection",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Optional detector model weights override",
    )
    return parser.parse_args()


def print_detection_summary(detections: Sequence[Detection]) -> None:
    """Print the sprint detection summary in the required format."""
    print("Detected Persons")
    print()

    if not detections:
        print("No persons detected.")
        return

    for index, detection in enumerate(detections, start=1):
        x1, y1, x2, y2 = detection["bbox"]

        print(f"Person {index}")
        print(f"Confidence: {detection['confidence']:.2f}")
        print("Bounding Box:")
        print(f"({x1},{y1})-({x2},{y2})")
        print()


def main() -> None:
    """Run the configured detection workflow using factory-based model creation."""
    configure_logging()
    args = parse_args()
    logger = logging.getLogger("guardian_vision.main")

    config = load_config(args.config)
    image_path_value = args.image if args.image else config.paths.image
    model_path_value = args.model if args.model else config.detector.model_path

    detector_config = DetectorConfig(
        model=config.detector.model,
        model_path=model_path_value,
        confidence_threshold=config.confidence_threshold,
        device=config.device,
        output_images_dir=config.paths.output_images,
        output_videos_dir=config.paths.output_videos,
    )
    detector = DetectionFactory.create(detector_config)

    image_path = Path(image_path_value)
    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    detections = detector.detect_image(str(image_path))

    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Failed to load image: {image_path}")

    output_image_name = f"{image_path.stem}_detected.jpg"
    output_image_path = detector.draw_detections(
        image=image,
        detections=detections,
        output_name=output_image_name,
    )

    print_detection_summary(detections)
    print(f"Saved annotated image to: {output_image_path}")

    if args.video:
        output_video_path = detector.detect_video(args.video)
        logger.info("Saved annotated video to: %s", output_video_path)


if __name__ == "__main__":
    main()
