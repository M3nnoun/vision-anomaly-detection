# Exploring Unsupervised Anomaly Detection Techniques on MVTec AD (Pill Dataset)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)

## Overview

This repository is a **personal learning project** where I explore and implement various **unsupervised anomaly detection** techniques on the **MVTec AD dataset**, focusing on the **Pill** category.

The goal is to experiment with different approaches for detecting defects in industrial images (e.g., cracks, contamination, color faults) using only normal ("good") samples for training. All methods follow the unsupervised paradigm: anomalies are detected based on deviation from the learned normal pattern.

This project is part of my ongoing learning process in computer vision and anomaly detection — a space to try new ideas, compare methods, and deepen understanding.

## Techniques Explored

I implemented and compared the following approaches:

1. **Simple Convolutional Autoencoder**
   - Classic reconstruction-based method.
   - A convolutional encoder-decoder is trained on normal images.
   - Anomaly score: Pixel-level L2 (MSE) reconstruction error.
   - Serves as a simple, interpretable baseline.

2. **Transfer Learning with ResNet50 + Deep Feature Reconstruction**
   - Inspired by the paper: *"Unsupervised Anomaly Segmentation via Deep Feature Reconstruction"* (Yang et al., Neurocomputing 2021, arXiv:2012.07122).
   - Extract multi-scale features from intermediate layers of a pretrained ResNet50 (e.g., layer2 and layer3 outputs).
   - Train a lightweight convolutional autoencoder to reconstruct these deep features (concatenated).
   - Anomaly score: Reconstruction error in feature space — often better for subtle anomalies than pixel-level reconstruction.

3. **KNN-based Anomaly Detection (Global / Image-Level)**
   - Extract global deep features from pretrained ResNet50 (e.g., average-pooled output).
   - Build a memory bank of feature vectors from normal training images.
   - Use K-Nearest Neighbors (KNN) distance to the closest normal features as the anomaly score.
   - Fast and effective for image-level detection.

4. **Patch-based KNN (Inspired by PatchCore)**
   - Based on *"Towards Total Recall in Industrial Anomaly Detection"* (Roth et al., CVPR 2022) — **PatchCore** by Amazon Science.
   - Extract patch-level features from intermediate ResNet50 layers.
   - Construct a compact memory bank of normal patches (using coreset subsampling for efficiency).
   - For test images, compare each patch to its nearest neighbors in the memory bank.
   - Anomaly score/map: Maximum or aggregated patch distances — enables precise anomaly localization.

## Dataset

- **MVTec AD - Pill Category**: Real-world industrial pill images.
  - Train: Only good/normal pills.
  - Test: Mix of good and defective pills (crack, color faults, contamination, etc.).
  - Download link provided in the notebook (or from official MVTec site).

## Requirements

```bash
pip install torch torchvision numpy matplotlib tqdm pillow scikit-learn faiss-gpu  # faiss for efficient KNN
```

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/mvtec-ad-anomaly-exploration.git
   cd mvtec-ad-anomaly-exploration
   ```

2. Open the Jupyter notebook(s) in Google Colab (recommended for GPU) or locally.

3. Run the cells sequentially:
   - Download and extract the dataset.
   - Implement/train each method.
   - Evaluate on the test set (image-level AUROC, pixel-level metrics where applicable).
   - Visualize reconstructions, anomaly heatmaps, and detections.

## Results & Learnings

- **Simple Autoencoder**: Easy to implement, good baseline, but struggles with very subtle defects.
- **Deep Feature Reconstruction**: Improved sensitivity and localization thanks to richer ResNet features.
- **Global KNN**: Quick image-level scoring with strong performance.
- **PatchCore-style**: Best localization, memory-efficient, close to state-of-the-art on MVTec AD.

Experimenting with these methods helped me understand the progression from basic reconstruction to modern memory-bank approaches.

## References

- Simple Autoencoder: Classic reconstruction-based anomaly detection.
- Deep Feature Reconstruction: [arXiv:2012.07122](https://arxiv.org/abs/2012.07122)
- PatchCore: [arXiv:2106.08265](https://arxiv.org/abs/2106.08265)  
  Official implementation: https://github.com/amazon-science/patchcore-inspection

## License

MIT License — feel free to fork, modify, and experiment!

## About

This is purely a learning and exploration project. I'm sharing it to document my progress and hopefully help others on the same journey.

Feedback, suggestions, or questions are very welcome! If you're interested in anomaly detection, computer vision, or industrial AI, let's connect.

Star ⭐ the repo if you found it useful! 🚀

