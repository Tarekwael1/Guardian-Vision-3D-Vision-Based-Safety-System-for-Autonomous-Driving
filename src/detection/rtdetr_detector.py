from __future__ import annotations

from typing import Sequence

import numpy as np

from src.detection.base import BaseDetector, Detection


class RTDETRDetector(BaseDetector):
    """Placeholder RT-DETR detector for future experiments."""

    def __init__(
        self,
        model_path: str,
        confidence_threshold: float,
        device: str,
    ) -> None:
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.device = device

    def detect_image(self, image_path: str) -> list[Detection]:
        raise NotImplementedError("RTDETRDetector is not implemented yet.")

    def draw_detections(
        self,
        image: np.ndarray,
        detections: Sequence[Detection],
        output_name: str | None = None,
    ) -> str:
        raise NotImplementedError("RTDETRDetector is not implemented yet.")

    def detect_video(self, video_path: str) -> str:
        raise NotImplementedError("RTDETRDetector is not implemented yet.")


class GroundingDINODetector(BaseDetector):
    """Placeholder GroundingDINO detector for future experiments."""

    def __init__(
        self,
        model_path: str,
        confidence_threshold: float,
        device: str,
    ) -> None:
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.device = device

    def detect_image(self, image_path: str) -> list[Detection]:
        raise NotImplementedError("GroundingDINODetector is not implemented yet.")

    def draw_detections(
        self,
        image: np.ndarray,
        detections: Sequence[Detection],
        output_name: str | None = None,
    ) -> str:
        raise NotImplementedError("GroundingDINODetector is not implemented yet.")

    def detect_video(self, video_path: str) -> str:
        raise NotImplementedError("GroundingDINODetector is not implemented yet.")
