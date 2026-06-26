from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from src.localization.models import PedestrianLocalization

LOGGER = logging.getLogger(__name__)


class Localizer:
    """Convert detections and depth maps into pedestrian localizations."""

    PERSON_CLASS_NAME = "person"

    def __init__(self, output_dir: str = "outputs/localization") -> None:
        self.output_dir = Path(output_dir)

    def localize(
        self,
        detections: Sequence[Mapping[str, Any]],
        depth_map: np.ndarray,
    ) -> list[PedestrianLocalization]:
        """Generate ordered pedestrian localizations from detections and depth."""
        if depth_map is None or depth_map.size == 0:
            raise ValueError("localize received an empty depth map")

        localized: list[PedestrianLocalization] = []

        for detection in detections:
            if str(detection.get("class_name", "")).lower() != self.PERSON_CLASS_NAME:
                continue

            bbox = self._parse_bbox(detection.get("bbox"))
            if bbox is None:
                LOGGER.warning("Skipping detection with invalid bbox: %s", detection)
                continue

            depth_values = self._extract_depth_values(depth_map, bbox)
            if depth_values.size == 0:
                LOGGER.warning("Skipping pedestrian with no valid depth values: %s", detection)
                continue

            x1, y1, x2, y2 = bbox
            localized.append(
                PedestrianLocalization(
                    id=len(localized) + 1,
                    class_name=str(detection.get("class_name", self.PERSON_CLASS_NAME)),
                    confidence=float(detection.get("confidence", 0.0)),
                    bbox=[x1, y1, x2, y2],
                    ground_point=((x1 + x2) // 2, y2),
                    relative_depth=float(np.median(depth_values)),
                )
            )

        localized.sort(key=lambda item: item.relative_depth)

        for index, item in enumerate(localized, start=1):
            item.id = index

        LOGGER.info("Generated %d pedestrian localization object(s)", len(localized))
        return localized

    def visualize(
        self,
        image: np.ndarray,
        localizations: Sequence[PedestrianLocalization],
        output_name: str | None = None,
    ) -> str:
        """Draw localization overlays and save the annotated image."""
        if image is None or image.size == 0:
            raise ValueError("visualize received an empty image")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_name = output_name or f"localization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        output_path = self.output_dir / output_name

        annotated = image.copy()
        for localization in localizations:
            x1, y1, x2, y2 = localization.bbox
            gx, gy = localization.ground_point

            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 180, 255), 2)
            cv2.circle(annotated, (gx, gy), 5, (0, 0, 255), -1)

            text_x = x1
            text_y = max(20, y1 - 35)
            cv2.putText(
                annotated,
                f"Person #{localization.id}",
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 180, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                annotated,
                f"Confidence: {localization.confidence:.2f}",
                (text_x, text_y + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 180, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                annotated,
                f"Relative Depth: {localization.relative_depth:.2f}",
                (text_x, text_y + 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 180, 255),
                2,
                cv2.LINE_AA,
            )

        if not cv2.imwrite(str(output_path), annotated):
            raise IOError(f"Failed to save localization visualization to {output_path}")

        LOGGER.info("Saved localization visualization to %s", output_path)
        return str(output_path)

    def _parse_bbox(self, bbox: Any) -> tuple[int, int, int, int] | None:
        """Validate and sanitize a detection bounding box."""
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            return None

        try:
            x1, y1, x2, y2 = (int(round(float(value))) for value in bbox)
        except (TypeError, ValueError):
            return None

        if x2 <= x1 or y2 <= y1:
            return None

        return x1, y1, x2, y2

    def _extract_depth_values(
        self,
        depth_map: np.ndarray,
        bbox: tuple[int, int, int, int],
    ) -> np.ndarray:
        """Apply the crop strategy and return cleaned depth values."""
        if depth_map.ndim != 2:
            raise ValueError("depth_map must be a 2D array")

        x1, y1, x2, y2 = bbox
        height, width = depth_map.shape[:2]

        if x1 >= width or y1 >= height or x2 <= 0 or y2 <= 0:
            return np.array([], dtype=np.float32)

        x1 = max(0, min(x1, width - 1))
        x2 = max(0, min(x2, width))
        y1 = max(0, min(y1, height - 1))
        y2 = max(0, min(y2, height))

        if x2 <= x1 or y2 <= y1:
            return np.array([], dtype=np.float32)

        bbox_width = x2 - x1
        bbox_height = y2 - y1

        left = x1 + int(round(0.10 * bbox_width))
        right = x2 - int(round(0.10 * bbox_width))
        top = y1 + int(round(0.10 * bbox_height))
        bottom = y2

        if right <= left or bottom <= top:
            return np.array([], dtype=np.float32)

        region_height = bottom - top
        use_top = max(top, bottom - int(round(0.30 * region_height)))

        if bottom <= use_top:
            return np.array([], dtype=np.float32)

        depth_region = depth_map[use_top:bottom, left:right]
        depth_values = depth_region[np.isfinite(depth_region)]
        if depth_values.size == 0:
            return np.array([], dtype=np.float32)

        depth_values = np.asarray(depth_values, dtype=np.float32).reshape(-1)
        depth_values.sort()

        trim_count = int(np.floor(depth_values.size * 0.05))
        if trim_count > 0 and depth_values.size > 2 * trim_count:
            depth_values = depth_values[trim_count:-trim_count]

        return depth_values
