from __future__ import annotations

import logging
import unittest

import numpy as np

from src.detection.yolo_detector import YOLODetector


class _FakeBox:
    def __init__(self, class_id: int, confidence: float, bbox: list[float]) -> None:
        self.cls = np.array([class_id], dtype=np.float32)
        self.conf = np.array([confidence], dtype=np.float32)
        self.xyxy = np.array([bbox], dtype=np.float32)


class _FakeResult:
    def __init__(self, boxes: list[_FakeBox]) -> None:
        self.names = {0: "person", 2: "car"}
        self.boxes = boxes


class TestYOLODetector(unittest.TestCase):
    def test_extract_person_only(self) -> None:
        detector = YOLODetector.__new__(YOLODetector)
        detector.logger = logging.getLogger("test_detector")
        detector.confidence_threshold = 0.0

        results = [
            _FakeResult(
                boxes=[
                    _FakeBox(class_id=0, confidence=0.91, bbox=[10.2, 12.8, 80.0, 140.4]),
                    _FakeBox(class_id=2, confidence=0.87, bbox=[1, 2, 3, 4]),
                ]
            )
        ]

        detections = detector._extract_person_detections(results)

        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0]["class_name"], "person")
        self.assertEqual(detections[0]["class_id"], 0)
        self.assertEqual(detections[0]["bbox"], [10, 13, 80, 140])


if __name__ == "__main__":
    unittest.main()
