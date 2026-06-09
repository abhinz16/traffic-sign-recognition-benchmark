# Traffic Sign Recognition and Cross-Dataset Robustness Benchmark

![Graphical Abstract](figures/graphical_abstract.png)

## Overview

This repository provides a large-scale traffic sign recognition benchmark built on multiple public datasets. The framework evaluates not only classification accuracy but also model robustness, cross-dataset generalization, and explainability.

The benchmark combines traffic sign imagery from GTSRB, BelgiumTS, and Mapillary whenever compatible label mappings are available. Models are trained using a unified 43-class GTSRB label space and evaluated using standard computer vision metrics together with robustness analyses.

## Key Features

* Combined multi-dataset training
* Automatic dataset acquisition
* CUDA acceleration with mixed precision
* Multi-core data loading
* ResNet18
* EfficientNet-B0
* Custom CNN baseline
* Confusion matrices
* Grad-CAM visualization
* Robustness testing
* Failure-case analysis

## Tested Environment

| Component | Version |
| --------- | ------- |
| Ubuntu    | 24.04   |
| Python    | 3.12    |
| PyTorch   | 2.x     |
| CUDA      | 12.x    |

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

Run:

```bash
python main.py
```

The pipeline automatically:

1. Downloads datasets where possible
2. Creates the combined training dataset
3. Trains the selected model
4. Evaluates performance
5. Generates visualizations
6. Saves benchmark reports

## Benchmark Tasks

* Single-dataset training
* Combined-dataset training
* Cross-dataset evaluation
* Robustness evaluation
* Explainability analysis

## Supported Datasets

| Dataset   | Purpose                  |
| --------- | ------------------------ |
| GTSRB     | Primary benchmark        |
| BelgiumTS | Cross-country validation |
| Mapillary | Real-world robustness    |

## Results

| Model           | Accuracy | F1 Score | Robustness Score |
| --------------- | -------- | -------- | ---------------- |
| ResNet18        | TBD      | TBD      | TBD              |
| EfficientNet-B0 | TBD      | TBD      | TBD              |
| Custom CNN      | TBD      | TBD      | TBD              |

## Outputs

```text
figures/
├── accuracy_curve.png
├── loss_curve.png
├── confusion_matrix.png
├── robustness_comparison.png
└── gradcam_examples.png

outputs/
├── classification_report.txt
├── training_history.csv
├── robustness_results.csv
└── failure_cases/
```

## Troubleshooting

### CUDA not detected

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

### BelgiumTS or Mapillary download fails

Place datasets manually:

```text
data/belgium/
data/mapillary/
```

### Out-of-memory errors

Reduce batch size in:

```python
config.py
```

## Future Work

* Vision Transformer support
* ONNX export
* Real-time inference benchmark
* Domain adaptation methods
