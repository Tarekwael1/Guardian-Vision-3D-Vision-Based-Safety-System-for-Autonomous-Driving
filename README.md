# Guardian Vision

**Guardian Vision** is a computer vision module for autonomous driving systems that enables a vehicle to understand pedestrian risk using a single RGB camera. Instead of relying on expensive sensors such as LiDAR, Guardian Vision combines human detection and monocular depth estimation to build a 3D understanding of the scene and estimate the risk posed by nearby pedestrians.

The project is designed as a perception module within a larger autonomous driving system. It does not directly control the vehicle; instead, it provides risk information for use by planning and control modules.

---

## Project Objectives

* Detect pedestrians in real time.
* Estimate the distance of each pedestrian from a monocular RGB image.
* Build a 3D representation of pedestrian locations.
* Classify pedestrians into different risk levels.
* Output risk information that autonomous driving decision systems can use.
* Evaluate the proposed approach using public datasets and a custom simulation dataset.

---

## System Pipeline

```text
RGB Camera
      │
      ▼
Pedestrian Detection
      │
      ▼
Monocular Depth Estimation
      │
      ▼
3D Pedestrian Localization
      │
      ▼
Guardian Risk Engine
      │
      ▼
Risk Assessment
      │
      ▼
Output to Planning Module
```

---

## Repository Structure (Planned)

```
GuardianVision/
│
├── datasets/
│   ├── raw/
│   ├── processed/
│   └── simulation/
│
├── docs/
│   ├── literature_review/
│   ├── methodology/
│   └── report/
│
├── models/
│   ├── detection/
│   ├── depth/
│   └── risk_engine/
│
├── notebooks/
│
├── scripts/
│
├── src/
│   ├── detection/
│   ├── depth/
│   ├── localization/
│   ├── risk_assessment/
│   └── visualization/
│
├── evaluation/
│
├── results/
│
├── presentation/
│
└── README.md
```

---

## Planned Technologies

| Component               | Technology                                |
| ----------------------- | ----------------------------------------- |
| Programming Language    | Python                                    |
| Deep Learning Framework | PyTorch                                   |
| Object Detection        | YOLO                                      |
| Depth Estimation        | Depth Anything                            |
| Simulation              | CARLA (planned)                           |
| Visualization           | OpenCV                                    |
| Experiment Tracking     | TensorBoard / Weights & Biases (optional) |

---

## Current Development Status

* [x] Project idea defined
* [x] Literature review started
* [x] Initial system architecture designed
* [ ] Repository structure finalized
* [ ] Baseline object detection
* [ ] Baseline depth estimation
* [ ] Integration of detection and depth
* [ ] Guardian Risk Engine
* [ ] Simulation dataset generation
* [ ] Training and evaluation
* [ ] Final report
* [ ] Final presentation

---

## Research Contribution

Guardian Vision aims to bridge the gap between 2D perception and 3D pedestrian risk assessment using only a monocular RGB camera.

The project combines:

* Real-time pedestrian detection
* Monocular depth estimation
* 3D pedestrian localization
* A novel risk assessment module (Guardian Risk Engine)

The primary research contribution is the design and evaluation of a vision-based risk assessment framework that classifies pedestrians into different safety levels based on their spatial relationship to the vehicle.

---

## Future Work

Potential extensions include:

* Multi-camera perception
* LiDAR and radar sensor fusion
* Pedestrian trajectory prediction
* Time-to-Collision estimation
* Vehicle speed-aware risk assessment
* Real-time deployment on embedded hardware

---

## References

The project is based on recent research in:

* Object Detection
* Monocular Depth Estimation
* Bird's Eye View Perception
* Autonomous Driving Safety Systems

The complete literature review will be available in the `docs/literature_review/` directory.

---

## License

This repository is currently under development for academic research purposes.
