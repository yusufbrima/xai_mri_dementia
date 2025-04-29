# Explainable MRI Dementia Diagnosis (PyTorch Port)

A PyTorch-based, end-to-end pipeline for dementia diagnosis from brain MRI scans with explainable AI (XAI) support.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Features (Implemented)](#features-implemented)
3. [Prerequisites](#prerequisites)
4. [Installation](#installation)
5. [Directory Structure](#directory-structure)
6. [Getting Started](#getting-started)
7. [Existing Components](#existing-components)
8. [Extending the Pipeline](#extending-the-pipeline)
9. [Configuration](#configuration)
10. [Logging & Checkpoints](#logging--checkpoints)
11. [Handoff Notes](#handoff-notes)
12. [License](#license)

---

## Project Overview

For a detailed system design and specification, see the [project overview PDF](docs/explainable_ai_mri_dementia.pdf).

This repository ports an existing Keras-based dementia diagnosis pipeline to PyTorch, emphasizing modularity, reproducible experiments, and interpretability via XAI methods (e.g., Grad-CAM, LRP).

This repository ports an existing Keras-based dementia diagnosis pipeline to PyTorch, emphasizing modularity, reproducible experiments, and interpretability via XAI methods (e.g., Grad-CAM, LRP).


## Features (Implemented)

- **Data Preprocessing & Loader**: NIfTI-based MRI preprocessing and dataset splits (`train`/`val`/`test`).
- **Model Definitions**: Three 3D CNN architectures already implemented under `models/`.
- **Training Utilities**: `train` function with support for configurable loss, optimizer, and metrics.
- **Model Persistence**: `save_model` and `load_model` functions for checkpointing and resume.


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
xai_mri_dementia/               # Repo root
├── config/                     # Experiment configs (YAML/JSON)
│   └── base_config.yaml        # Default hyperparams & paths
├── data/                       # Preprocessing & dataset code
│   ├── transforms.py           # Image transforms & augmentations
│   ├── nifti_loader.py         # NIfTI MRI loader
│   └── dataset_factory.py      # Builds Train/Val/Test datasets
├── models/                     # Neural network architectures
│   ├── base_model.py           # Abstract base class
│   ├── cnn_backbones.py        # 3D CNN variants
│   └── model_factory.py        # Instantiate models via config
├── explainers/                 # XAI methods (e.g., LRP, Saliency)
│   ├── lrp.py
│   ├── vanilla.py
│   └── explainer_factory.py
├── training/                   # Training & evaluation logic
│   ├── trainer.py              # Main training loop
│   ├── losses.py
│   └── metrics.py
├── utils/                      # Utility functions
│   ├── logger.py               # Experiment logging
│   ├── visualization.py        # Plotting and result visualization
│   └── nifti_utils.py          # Helpers for NIfTI IO
├── scripts/                    # CLI entrypoints
│   └── train.py                # Launch training from command line
├── results/                    # Experiment outputs (logs, plots, checkpoints)
│   └── exp_<n>/
├── tests/                      # (Removed – unit tests not in use)
├── requirements.txt
└── README.md
```  


## Getting Started

### Quick Training Example

```bash
python scripts/train.py \
  --config config/base_config.yaml \
  --experiment_name exp01
```

This will:
1. Load MRI data via `data/dataset_factory.py`.
2. Initialize the model (one of the three provided) via `models/model_factory.py`.
3. Execute training loop in `training/trainer.py`.
4. Save checkpoints & logs under `results/exp01/`.


## Existing Components

- **Data Loader**: `data/nifti_loader.py` and `data/transforms.py` handle raw NIfTI files and augmentations.
- **Models**: Three 3D CNNs (`models/cnn_backbones.py`), abstracted by `BaseModel` and assembled in `model_factory.py`.
- **Train / Save / Load**:
  - `training/trainer.py`: core training logic with loss, optimizer configuration.
  - `utils/logger.py`: logging of metrics and checkpoints.
  - `models/base_model.py`: includes `save_model` & `load_model` methods.


## Extending the Pipeline

### Adding a New Model
1. Define your model class in `models/` (e.g., `custom_model.py`), subclassing `BaseModel`.
2. Register it in `models/model_factory.py` with a unique name.
3. Add a corresponding entry under `model_name` in your config file.

### Implementing a New XAI Method
1. Create a new script under `explainers/` (e.g., `gradcam.py`).
2. Add a factory entry in `explainer_factory.py`.
3. Invoke via CLI or notebook, ensuring correct `model` and `data` inputs.


## Configuration

All hyperparameters, paths, and component choices live in YAML/JSON under `config/`.  
Modify `base_config.yaml` or create new ones, then pass via `--config` flag.

Example fields:
```yaml
model_name: resnet3d
batch_size: 8
learning_rate: 1e-4
data_root: /path/to/nifti/
explainer: lrp
```


## Logging & Checkpoints

- Logs (e.g., training & validation metrics) are saved in `<results>/logs/`.
- Model weights and optimizer state are checkpointed via `save_model` in `<results>/checkpoints/`.
- Resume training by pointing `load_model` to an existing checkpoint in your config.


## Handoff Notes

- Activate the `xai_dementia` Conda environment before running any scripts.
- Update `requirements.txt` when adding new dependencies:
  ```bash
  pip freeze > requirements.txt
  ```
- Use descriptive experiment names to avoid overwriting previous runs.
- For any missing functionality (e.g., additional explainers), follow the existing factory patterns.


## License

This project is released under the [MIT License](LICENSE).