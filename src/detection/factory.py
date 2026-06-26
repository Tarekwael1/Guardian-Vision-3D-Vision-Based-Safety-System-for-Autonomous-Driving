from __future__ import annotations

from dataclasses import dataclass

from src.detection.base import BaseDetector
from src.detection.rtdetr_detector import GroundingDINODetector, RTDETRDetector
from src.detection.yolo_detector import YOLODetector


@dataclass(frozen=True)
class DetectorConfig:
    """Detector runtime configuration used by DetectionFactory."""

    model: str = "yolo"
    model_path: str = "yolo11n.pt"
    confidence_threshold: float = 0.0
    device: str = "cpu"
    output_images_dir: str = "outputs/images"
    output_videos_dir: str = "outputs/videos"


class DetectionFactory:
    """Factory for constructing detector implementations by name."""

    @staticmethod
    def create(config: DetectorConfig) -> BaseDetector:
        model_name = config.model.strip().lower()

        if model_name == "yolo":
            return YOLODetector(
                model_path=config.model_path,
                confidence_threshold=config.confidence_threshold,
                device=config.device,
                output_images_dir=config.output_images_dir,
                output_videos_dir=config.output_videos_dir,
            )

        if model_name == "rtdetr":
            return RTDETRDetector(
                model_path=config.model_path,
                confidence_threshold=config.confidence_threshold,
                device=config.device,
            )

        if model_name in {"groundingdino", "grounding_dino"}:
            return GroundingDINODetector(
                model_path=config.model_path,
                confidence_threshold=config.confidence_threshold,
                device=config.device,
            )

        raise ValueError(f"Unsupported detector model: {config.model}")
