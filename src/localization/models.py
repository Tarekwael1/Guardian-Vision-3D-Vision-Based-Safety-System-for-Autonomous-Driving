from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class PedestrianLocalization:
    """Structured localization output for a single pedestrian."""

    id: int
    class_name: str
    confidence: float
    bbox: list[int]
    ground_point: tuple[int, int]
    relative_depth: float
    metric_distance: float | None = None
    risk_score: float | None = None
    risk_level: str | None = None
