from __future__ import annotations

import numpy as np

from src.depth.base import BaseDepthEstimator


class Metric3DEstimator(BaseDepthEstimator):
    """Placeholder Metric3D estimator for future experiments."""

    def __init__(self, model_path: str, device: str) -> None:
        self.model_path = model_path
        self.device = device

    def estimate_image(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Metric3DEstimator is not implemented yet.")
