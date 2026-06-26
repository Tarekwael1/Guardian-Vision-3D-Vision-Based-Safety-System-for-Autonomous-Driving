from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import numpy as np

from src.detection.base import Detection
from src.utils.image_utils import (
    create_output_filename,
    draw_person_detections,
    ensure_output_dir,
    save_image,
)


class Visualizer:
    """Visualization helper for drawing and saving detection artifacts."""

    def __init__(self, image_output_dir: Path) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.image_output_dir = image_output_dir

    def draw_on_image(self, image: np.ndarray, detections: Sequence[Detection]) -> np.ndarray:
        """Draw detections on an in-memory image and return the annotated copy."""
        if image is None or image.size == 0:
            raise ValueError("draw_on_image received an empty image")
        return draw_person_detections(image=image, detections=detections)

    def draw_and_save_image(
        self,
        image: np.ndarray,
        detections: Sequence[Detection],
        output_name: str | None = None,
    ) -> str:
        """Draw detections and save the image to the configured output folder."""
        output_dir = ensure_output_dir(self.image_output_dir)
        final_name = output_name or create_output_filename("detected", ".jpg")
        output_path = output_dir / final_name

        annotated = self.draw_on_image(image=image, detections=detections)
        save_image(annotated, output_path)

        self.logger.info("Annotated image saved to %s", output_path)
        return str(output_path)
