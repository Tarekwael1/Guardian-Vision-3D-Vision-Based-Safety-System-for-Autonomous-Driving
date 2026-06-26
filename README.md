# Guardian Vision - Modular Research Framework

Guardian Vision is a master's research framework for pedestrian-centric scene understanding in autonomous driving. The current implementation preserves the existing human detection behavior and refactors the architecture to support multiple detectors, multiple depth models, and future localization, risk, and evaluation modules.

## Current Status

- Human detection is fully functional with YOLO through a factory-based interface.
- Depth, localization, risk, and evaluation modules are scaffolded for future sprints.
- Placeholder model classes intentionally raise `NotImplementedError`.

## Project Structure

```text
GuardianVision/

src/
├── config/
│   ├── __init__.py
│   ├── config.yaml
│   └── loader.py
├── depth/
│   ├── __init__.py
│   ├── base.py
│   ├── factory.py
│   ├── depth_anything.py
│   ├── metric3d.py
│   ├── midas.py
│   └── test_depth.py
├── detection/
│   ├── __init__.py
│   ├── base.py
│   ├── factory.py
│   ├── yolo_detector.py
│   ├── rtdetr_detector.py
│   └── test_detector.py
├── evaluation/
│   └── __init__.py
├── localization/
│   ├── __init__.py
│   └── localizer.py
├── risk/
│   └── __init__.py
├── utils/
│   ├── __init__.py
│   └── image_utils.py
├── visualization/
│   ├── __init__.py
│   └── visualizer.py
└── main.py

datasets/
models/
outputs/
├── images/
└── videos/

tests/
requirements.txt
main.py
README.md
```

## Design Principles Applied

- Abstract Base Classes (`BaseDetector`, `BaseDepthEstimator`)
- Factory Pattern (`DetectionFactory`, `DepthFactory`)
- Dataclass-backed configuration objects
- Strong typing and docstrings
- Logging-based execution tracing
- Separation of inference and visualization responsibilities

## Configuration

Runtime settings are loaded from [src/config/config.yaml](src/config/config.yaml).

Example:

```yaml
detector:
    model: yolo
    model_path: yolo11n.pt

depth:
    model: depth_anything
    model_path: ""

device: cpu
confidence_threshold: 0.0

paths:
    image: ../street_scene.jpg
    output_images: outputs/images
    output_videos: outputs/videos
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run with YAML configuration:

```bash
python main.py
```

Override image or detector weights from CLI:

```bash
python main.py --image datasets/000098.png --model yolo11n.pt
```

Optional video processing:

```bash
python main.py --image datasets/000098.png --video path/to/video.mp4
```

`main.py` at the project root is kept as a compatibility wrapper and delegates to `src.main`.

## Detection Output Schema

Person detections use a reusable, module-agnostic schema:

```python
{
        "class_id": int,
        "class_name": str,
        "confidence": float,
        "bbox": [x1, y1, x2, y2],
}
```

This format is intended for downstream depth, localization, risk, and evaluation modules.

## Placeholder Modules

The following classes are scaffolds and intentionally not implemented yet:

- `RTDETRDetector`
- `GroundingDINODetector`
- `DepthAnythingEstimator`
- `Metric3DEstimator`
- `MiDaSEstimator`
- `ZoeDepthEstimator`
- `Localizer`

## Tests

```bash
python -m unittest src.detection.test_detector
python -m unittest src.depth.test_depth
```
