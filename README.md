# scratch-DETR

A PyTorch project for training DETR (DEtection TRansformer) on a custom dataset, built for learning and experimentation.

## Features

- 🚀 **PyTorch Lightning** for scalable and organized training loops.
- 📊 **Custom Dataset Support**: Built-in COCO-format dataset handling and preprocessing.
- 🖼️ **Visualization**: Utilities to visualize model predictions on your images.

## Installation

### Prerequisites

- Python >= 3.11
- CUDA-compatible GPU (recommended)
- [uv](https://docs.astral.sh/uv/) package manager (recommended)

### Quick Install with uv (Recommended)

```bash
# Install dependencies from the project root
uv sync
```

## Usage

### 1. Data Preparation

This project uses the COCO object detection format. Place your dataset inside the `data/` directory. The script expects the following structure:

**Dataset Structure:**
```
data/
└── coco_lp/
    ├── train/
    │   ├── custom_train.json
    │   └── ...images
    └── val/
        ├── custom_val.json
        └── ...images
```
*Note: The data directory is gitignored.*

### 2. Training

You can train the DETR model using the provided script.

```bash
# Using uv
uv run python scripts/train_detr.py --data_dir data/coco_lp --accelerator gpu --batch_size 4
```
*CLI arguments can override defaults found in `src/config.py`.*

### 3. Monitoring with TensorBoard

Monitor training progress using TensorBoard.

```bash
# Point TensorBoard to the logs directory
tensorboard --logdir lightning_logs
```

### 4. Jupyter Notebooks

Explore the notebooks for a deeper dive into the model and data.

```bash
# Start Jupyter Lab
uv run jupyter lab

# Open notebooks in the notes/ directory
```

## Project Structure

```
scratch-DETR/
├── data/                    # Dataset storage (gitignored)
├── lightning_logs/          # Training logs and checkpoints (gitignored)
├── notes/                   # Jupyter notebooks and images
├── scripts/                 # Training script
│   └── train_detr.py
├── src/                     # Source code
│   ├── config.py            # Training configurations
│   ├── data.py              # Dataloader for baseline DETR
│   ├── model.py             # Lightning Module for baseline DETR
│   └── utils.py             # Visualization helpers
└── pyproject.toml           # Project dependencies
```

## License

This repository is for research and educational purposes. Please check the original licenses for the models and datasets before any commercial use.

