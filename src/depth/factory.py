from __future__ import annotations

from dataclasses import dataclass

from src.depth.base import BaseDepthEstimator
from src.depth.depth_anything import DepthAnythingEstimator


@dataclass(frozen=True)
class DepthConfig:
    """Depth runtime configuration used by DepthFactory."""

    model: str = "depth_anything"
    model_path: str = ""
    device: str = "cpu"
    output_dir: str = "outputs/depth"


class _NotImplementedDepthEstimator(BaseDepthEstimator):
    """Placeholder estimator used for planned but unimplemented models."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def load_model(self) -> None:
        raise NotImplementedError(f"{self.model_name} is not implemented yet.")

    def estimate_depth(self, image) -> object:
        raise NotImplementedError(f"{self.model_name} is not implemented yet.")

    def visualize(self, depth_map) -> str:
        raise NotImplementedError(f"{self.model_name} is not implemented yet.")


class DepthFactory:
    """Factory for constructing depth estimator implementations by name."""

    @staticmethod
    def create(config: DepthConfig) -> BaseDepthEstimator:
        model_name = config.model.strip().lower()

        if model_name in {"depth_anything", "depthanything"}:
            return DepthAnythingEstimator(
                model_path=config.model_path,
                device=config.device,
                output_dir=config.output_dir,
            )

        if model_name in {"metric3d", "metric_3d"}:
            return _NotImplementedDepthEstimator(model_name="Metric3D")

        if model_name == "midas":
            return _NotImplementedDepthEstimator(model_name="MiDaS")

        if model_name in {"zoedepth", "zoe_depth"}:
            return _NotImplementedDepthEstimator(model_name="ZoeDepth")

        raise ValueError(f"Unsupported depth model: {config.model}")
