from __future__ import annotations

import numpy as np

from src.depth.base import BaseDepthEstimator


class MiDaSEstimator(BaseDepthEstimator):
    """Placeholder MiDaS estimator for future experiments."""

    def __init__(self, model_path: str, device: str) -> None:
        self.model_path = model_path
        self.device = device

    def estimate_image(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError("MiDaSEstimator is not implemented yet.")


class ZoeDepthEstimator(BaseDepthEstimator):
    """Placeholder ZoeDepth estimator for future experiments."""

    def __init__(self, model_path: str, device: str) -> None:
        self.model_path = model_path
        self.device = device

    def estimate_image(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError("ZoeDepthEstimator is not implemented yet.")
