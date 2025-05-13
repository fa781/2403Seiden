# Video Frame Selection and Object Detection Pipeline

This repository implements a modular pipeline for selecting informative frames from a video and performing object detection using Grounding DINO.

## Features

- **Input Processing**: Extract and sample I-frames from input videos.
- **Zero-Shot Object Detection**: Run detection with Grounding DINO on sampled frames.
- **MAB Sampling**: Iteratively select frames using a multi-armed bandit approach.
- **Label Propagation**: Interpolate detection scores across all frames.
- **AQP Queries**: Perform approximate query processing for frame selection.
- **Visualization**: Output annotated frames with bounding boxes and scores.

## Getting Started

Run the full pipeline with:
```bash
python main.py
