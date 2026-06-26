from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DetectorSettings:
    model: str
    model_path: str


@dataclass(frozen=True)
class DepthSettings:
    model: str
    model_path: str


@dataclass(frozen=True)
class PathSettings:
    image: str
    output_images: str
    output_videos: str


@dataclass(frozen=True)
class AppConfig:
    detector: DetectorSettings
    depth: DepthSettings
    paths: PathSettings
    device: str
    confidence_threshold: float


def load_config(config_path: str | Path) -> AppConfig:
    """Load application settings from a YAML configuration file."""
    resolved_path = Path(config_path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"Config file not found: {resolved_path}")

    with resolved_path.open("r", encoding="utf-8") as file:
        raw: dict[str, Any] = yaml.safe_load(file) or {}

    detector_raw = raw.get("detector", {})
    depth_raw = raw.get("depth", {})
    paths_raw = raw.get("paths", {})

    return AppConfig(
        detector=DetectorSettings(
            model=str(detector_raw.get("model", "yolo")),
            model_path=str(detector_raw.get("model_path", "yolo11n.pt")),
        ),
        depth=DepthSettings(
            model=str(depth_raw.get("model", "depth_anything")),
            model_path=str(depth_raw.get("model_path", "")),
        ),
        paths=PathSettings(
            image=str(paths_raw.get("image", "../street_scene.jpg")),
            output_images=str(paths_raw.get("output_images", "outputs/images")),
            output_videos=str(paths_raw.get("output_videos", "outputs/videos")),
        ),
        device=str(raw.get("device", "cpu")),
        confidence_threshold=float(raw.get("confidence_threshold", 0.5)),
    )
