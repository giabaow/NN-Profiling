# Neural Network GPU/CPU Profiling

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.1%2B-orange)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

This repository demonstrates **profiling memory and compute usage** of different neural network architectures (MLP, CNN, Transformer block) using **PyTorch**. It is designed to run on **Mac MPS** (Apple Silicon) or CPU.

---

## Features

- Implements three architectures:
  - **MLP** (Multi-Layer Perceptron)
  - **Simple CNN**
  - **Transformer Block**
- Generates **synthetic data** with varying batch sizes
- Profiles **CPU time** and **GPU memory usage** (via MPS memory allocation)
- Saves **memory and compute plots** to `results/` folder

---

## Setup

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/NN-Profiling.git
cd NN-Profiling
