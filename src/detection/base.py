from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Sequence, TypedDict

import numpy as np


class Detection(TypedDict):
    """Generic detection payload shared across framework modules."""

    class_id: int
    class_name: str
    confidence: float
    bbox: List[int]


class BaseDetector(ABC):
    """Abstract detector interface for pluggable detection backends."""

    @abstractmethod
    def detect_image(self, image_path: str) -> List[Detection]:
        """Run inference on a single image and return detections."""

    @abstractmethod
    def draw_detections(
        self,
        image: np.ndarray,
        detections: Sequence[Detection],
        output_name: str | None = None,
    ) -> str:
        """Draw detections and persist an annotated image."""

    @abstractmethod
    def detect_video(self, video_path: str) -> str:
        """Run inference on a video and persist an annotated output video."""
