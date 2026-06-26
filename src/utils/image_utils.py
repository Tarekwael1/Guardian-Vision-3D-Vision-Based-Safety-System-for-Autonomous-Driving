from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)


def ensure_output_dir(directory: Path) -> Path:
    """Create an output directory if it does not exist."""
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def create_output_filename(prefix: str, extension: str) -> str:
    """Create a timestamped output file name."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    extension = extension if extension.startswith(".") else f".{extension}"
    return f"{prefix}_{timestamp}{extension}"


def draw_person_detections(
    image: np.ndarray,
    detections: Sequence[Mapping[str, Any]],
) -> np.ndarray:
    """Draw person detections on a copy of the provided image."""
    annotated = image.copy()

    for detection in detections:
        x1, y1, x2, y2 = detection["bbox"]
        confidence = float(detection["confidence"])

        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 220, 0), 2)
        label = f"Person {confidence:.2f}"
        cv2.putText(
            annotated,
            label,
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 220, 0),
            2,
            cv2.LINE_AA,
        )

    return annotated


def save_image(image: np.ndarray, output_path: Path) -> None:
    """Persist an image and raise an error if writing fails."""
    if not cv2.imwrite(str(output_path), image):
        raise IOError(f"Failed to save output image to: {output_path}")
    LOGGER.info("Image saved to %s", output_path)
