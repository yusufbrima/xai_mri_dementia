Explainable MRI Dementia Diagnosis (PyTorch Port)

A PyTorch-based, end-to-end port of the original Keras pipeline ([martindyrba/Experimental](https://github.com/martindyrba/Experimental)) for dementia diagnosis from brain MRI scans with explainable AI (XAI) support.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Features (Implemented)](#features-implemented)
3. [Prerequisites](#prerequisites)
4. [Installation](#installation)
5. [Directory Structure](#directory-structure)
6. [Getting Started](#getting-started)
7. [Examples](#examples)
8. [Existing Components](#existing-components)
9. [Extending the Pipeline](#extending-the-pipeline)
10. [Configuration](#configuration)
11. [Logging & Checkpoints](#logging--checkpoints)
12. [Handoff Notes](#handoff-notes)
13. [License](#license)

---

## Project Overview

For a detailed system design and specification, see the [project overview PDF](docs/explainable_ai_mri_dementia.pdf).

This repository ports an existing Keras-based dementia diagnosis pipeline to PyTorch, focusing on modularity, reproducible experiments, and interpretability via XAI methods (e.g., Grad-CAM, LRP).

## Features (Implemented)

- **YAML Configuration**: Base experiment configuration (`config/base_config.yaml`) fully supported.
- **Data Preprocessing & Loader**: NIfTI-based MRI preprocessing, augmentations, and dataset splits (`train`/`val`/`test`) implemented in `data/nifti_loader.py` and `data/transforms.py`.
- **Initial 3D CNN Models**: Simple3DCNN and ResNet3D architectures provided under `models/` as inspiration and reference.
- **Training Utilities**:
  - `train_model(...)` in `training/trainer.py` for end-to-end classification (TODO: intermediate checkpointing).
  - `test_model(...)` for evaluation of trained models on test splits.
- **Model Persistence**:
  - `save_model()` and `load_model()` helpers for saving/loading weights and optimizer state.
- **CLI Interface**: `main.py` entrypoint supporting default config, custom config files, and command-line overrides for batch size, epochs, and other parameters.
- **Starter Notebook**: `examples/Starter.ipynb` demonstrating data loading, config overrides, data iteration, slice plotting, and full training workflow.

## Prerequisites

- Python ≥ 3.12
- Conda (Anaconda or Miniconda)
- CUDA Toolkit ≥ 12.6 (optional, for GPU acceleration)

It’s recommended to create and activate a dedicated Conda environment before proceeding.

## Installation

```bash
# Clone the repository
git clone https://github.com/yusufbrima/xai_mri_dementia.git
cd xai_mri_dementia

# Create and activate a Conda environment
conda create -n xai_dementia python=3.12 -y
conda activate xai_dementia

# Install dependencies
pip install -r requirements.txt
```

## Directory Structure

```text
explainable_mri_dementia/               # Repo root
├── config/                            # YAML configs and experiment initialization
│   ├── base_config.yaml               # Default hyperparameters & paths
│   └── base_loader.py                 # Config loader and experiment directory setup
├── data/                              # Data loading & preprocessing
│   ├── transforms.py                  # Image transformations & augmentations
│   └── nifti_loader.py                # NIfTI MRI file loader
├── models/                            # Neural network architectures
│   └── cnn_backbones.py               # Simple3DCNN & ResNet3D variants
├── explainers/                        # XAI methods for interpretability
│   ├── lrp.py                         # Layer-wise Relevance Propagation
│   └── vanilla.py                     # Vanilla gradient saliency maps
├── training/                          # Training and evaluation logic
│   ├── trainer.py                     # Main training loop (`train_model`, `test_model`)
│   ├── losses.py                      # Custom loss functions
│   └── metrics.py                     # Evaluation metrics
├── utils/                             # Miscellaneous utilities
│   ├── logger.py                      # Logging utilities
│   ├── visualization.py               # Plotting results and saliency maps
│   └── utils.py                       # General helper functions
├── examples/                          # Example notebooks and demos
│   └── Starter.ipynb                  # Quickstart notebook
├── results/                           # Experiment outputs
│   ├── exp_01/                        # Outputs for experiment 01
│   └── exp_02/                        # Outputs for experiment 02
├── main.py                            # Entry point for CLI usage
├── requirements.txt                   # Python dependencies
└── README.md                          # Overview and usage instructions
```

## Getting Started

### Quick Training via CLI

```bash
python scripts/train.py \
  --config config/base_config.yaml \
  --experiment_name exp01
```

This will:
1. Load MRI data via `data/nifti_loader.py`.
2. Initialize your chosen model via `models/cnn_backbones.py`.
3. Execute the training loop in `training/trainer.py`.
4. Save metrics, checkpoints, and logs under `results/exp01/` including the current date (DDMMYYYY) affixed to it.

## Usage Examples

1. **Run with default config:**  
   `python main.py`
2. **Specify a custom config file:**  
   `python main.py -c config/base_config.yaml`
3. **Override batch size and epochs:**  
   `python main.py --batch-size 16 --epochs 5`
4. **Use generic overrides:**  
   `python main.py -o data.batch_size=32 -o training.epochs=3`
5. **Combine both:**  
   `python main.py -c config/base_config.yaml --batch-size 8 -o training.lr=1e-4`

## Jupter Notebook Examples

Launch the starter notebook for interactive exploration:

```bash
jupyter notebook examples/Starter.ipynb
```

The notebook covers:
- Loading default and overridden configs.
- Building and iterating data loaders.
- Visualizing MRI slices and transforms.
- Running the training and evaluation pipeline.

## Existing Components

- **Data Loader**: Complete NIfTI handling and splits (`data/nifti_loader.py`, `data/transforms.py`).
- **Models**: Simple3DCNN & ResNet3D in `models/cnn_backbones.py`.
- **Training & Evaluation**:
  - `train_model(...)` & `test_model(...)` in `training/trainer.py`.
  - `save_model()` / `load_model()` in `models/base_model.py` and `utils/logger.py`.

## Extending the Pipeline

### Adding a New Model
1. Subclass `nn.Module` in `models/cnn_backbones.py`.
2. Add entry under `model_name` in your YAML config.

### Implementing a New XAI Method
1. Create script in `explainers/` (e.g., `gradcam.py`).
2. Invoke via CLI or notebook with correct `--explainer` flag.

## Configuration

All settings for hyperparameters, paths, and components live in `config/base_config.yaml`. Override defaults by passing additional flags or custom config files:

```yaml
model_name: resnet3d
batch_size: 8
learning_rate: 1e-4
data_root: /path/to/nifti/
explainer: lrp
```

## Logging & Checkpoints

- Metrics and logs: `results/<exp>/logs/`.
- Checkpoints (model & optimizer state): `results/<exp>/checkpoints/` via `save_model()`.
- To evaluate on test split, use `test_model()` which outputs test metrics and optionally save results.

## Handoff Notes

- Activate `xai_dementia` environment before running any scripts.
- After adding new dependencies, update `requirements.txt`:
  ```bash
  pip freeze > requirements.txt
  ```
- Use descriptive experiment names to avoid overwriting previous results.
- Refer to `examples/Starter.ipynb` for usage patterns.

## License

This project is released under the [MIT License](LICENSE).