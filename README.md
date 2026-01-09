# KoLeo Loss Study

This repository contains the code and experiments for studying the KoLeo loss regularization technique in siamese networks trained with triplet loss on CIFAR-10.

This work is part of a series of articles exploring:
- [Training a siamese network with a triplet loss on CIFAR-10 - Part 1/3](https://adlane.me/blogs/chapter1-training-siamese-network/)
- [Effect of KoLeo loss on triplet loss - Part 2/3](https://adlane.me/blogs/chapter2-koleo-loss/)
- [Gradient accumulation and KoLeo loss - Part 3/3](https://adlane.me/blogs/chapter3-gradient-accumulation/)

## Overview

The KoLeo loss is a batch-dependent regularization technique that encourages embeddings to spread uniformly in the representation space by maximizing the minimum distance between them. This study investigates:

- The effect of KoLeo loss on embedding distribution
- The impact of gradient accumulation on batch-dependent losses
- Hyperparameter optimization for optimal model performance

## Setup

### Prerequisites

- Python 3.8+
- CIFAR-10 dataset (download and extract to `../cifar-10-python/`)

### Installation

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

2. Download and prepare the CIFAR-10 dataset:

The dataset should be extracted to `../cifar-10-python/` relative to the project root. The directory should contain the following files:
- `data_batch_1` through `data_batch_5`
- `test_batch`
- `batches.meta`

## Hyperparameter Search

The hyperparameter search uses [Optuna](https://optuna.org/) to systematically explore the hyperparameter space and find optimal configurations for training siamese networks with triplet loss and KoLeo regularization.

### Running the Hyperparameter Search

To run the hyperparameter search, execute:

```bash
python hp_search.py
```

### Search Space

The hyperparameter search explores the following parameters:

- **Triplet loss margin**: Float between 0.2 and 0.6
- **KoLeo loss usage**: Boolean (enable/disable KoLeo loss)
- **KoLeo loss weight**: Log-uniform distribution between 1e-4 and 1e-1 (only when KoLeo is enabled)
- **Gradient accumulation steps**: Categorical [1, 2, 4]
- **Batch size**: Categorical [64, 128]
- **Learning rate**: Log-uniform distribution between 1e-4 and 1e-2
- **Embedding size**: Categorical [64, 128, 256]
- **Optimizer**: Categorical ["Adam", "RMSprop"]

### Best Results

After running 100 trials, the best configuration achieved:

- **Best AUC**: 0.9597
- **Best hyperparameters**:
  - Triplet loss margin: 0.265
  - Use KoLeo loss: False
  - Gradient accumulation steps: 1
  - Batch size: 64
  - Learning rate: 0.000161
  - Embedding size: 128
  - Optimizer: Adam

Interestingly, the best result was achieved without KoLeo loss regularization, suggesting that for this particular task and dataset, the triplet loss alone with carefully tuned hyperparameters provides optimal performance.

### Visualizing Results

To interactively explore the hyperparameter search results, you can use the Optuna dashboard:


```bash
optuna-dashboard sqlite:///optuna_study.db
```

This will start a web server (typically at `http://127.0.0.1:8080`).

The dashboard provides an interactive interface to explore all 100 trials and understand how different hyperparameters affect model performance.
