from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseDepthEstimator(ABC):
    """Abstract interface for pluggable monocular depth estimators."""

    @abstractmethod
    def load_model(self) -> None:
        """Load model weights and initialize inference components."""

    @abstractmethod
    def estimate_depth(self, image: np.ndarray) -> np.ndarray:
        """Estimate a dense depth map from an input RGB/BGR image."""

    @abstractmethod
    def visualize(self, depth_map: np.ndarray) -> str:
        """Save a colorized visualization for a depth map and return its path."""
