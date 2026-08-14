# Unsupervised Anomaly Detection on MVTec AD (Pill Category)

Four unsupervised approaches to industrial defect detection, implemented from scratch and
benchmarked on the same data split — from a pixel-space autoencoder baseline up to a
PatchCore-style memory bank.

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

## Problem

In industrial quality control, defective samples are rare and their failure modes are open-ended,
so collecting a labelled defect dataset is impractical. The unsupervised framing sidesteps this:
train only on *known-good* parts, learn what "normal" looks like, and score test images by how far
they deviate from it.

This project applies that framing to the **Pill** category of
[MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad), where defects include cracks,
contamination, colour faults, and print errors.

## Approaches implemented

| # | Method | Anomaly score |
|---|--------|---------------|
| 1 | Convolutional autoencoder | Pixel-level L2 reconstruction error |
| 2 | ResNet50 deep-feature reconstruction | Reconstruction error in feature space |
| 3 | Global KNN on ResNet50 embeddings | Distance to nearest normal embedding |
| 4 | Patch-based KNN (PatchCore-style) | Max/aggregated patch distance to memory bank |

**1. Convolutional autoencoder.** An encoder-decoder trained on good images only. Anomalies are
assumed to reconstruct poorly because the decoder has never learned to represent them. Simple and
interpretable, and it doubles as the baseline everything else is measured against.

**2. ResNet50 + deep feature reconstruction.** Rather than reconstructing pixels, extract
multi-scale activations from intermediate ResNet50 layers (`layer2`, `layer3`) and train a light
autoencoder to reconstruct those. Following
[Yang et al., *Unsupervised Anomaly Segmentation via Deep Feature Reconstruction*](https://arxiv.org/abs/2012.07122)
(Neurocomputing 2021). Feature space is less sensitive to nuisance pixel variation, which matters
for subtle defects.

**3. Global KNN.** Average-pool ResNet50 features into one vector per image, store the training
vectors as a memory bank, and score a test image by its KNN distance to that bank. Cheap, no
training — but a single global vector dilutes small localised defects, which shows in the results.

**4. Patch-based KNN.** The PatchCore idea from
[Roth et al., *Towards Total Recall in Industrial Anomaly Detection*](https://arxiv.org/abs/2106.08265)
(CVPR 2022): keep patch-level features instead of one global vector, subsample them into a compact
coreset, and score each test patch against its nearest neighbours. Recovers the spatial precision
that approach 3 loses and yields a usable anomaly heatmap.

## Results

Image-level AUROC on the MVTec AD Pill test split (good vs. all defect types):

| Method | AUROC |
|--------|------:|
| Convolutional autoencoder | 0.817 |
| **ResNet50 deep feature reconstruction** | **0.943** |
| Global KNN | 0.685 |
| Patch-based KNN (PatchCore-style) | 0.923 |

Reading the numbers:

- **Pretrained features beat pixels.** The two methods built on ResNet50 features and spatial
  structure (0.943, 0.923) clearly outperform the pixel-space autoencoder (0.817).
- **Global pooling is the weak link.** Global KNN is the *worst* performer here (0.685), below even
  the plain autoencoder. Average-pooling the whole image into one vector washes out exactly the
  small, localised defects this dataset is made of — the same backbone used patch-wise scores 0.923.
- **Deep feature reconstruction edges out patch KNN** on image-level AUROC, though patch KNN gives
  noticeably better defect *localisation* in the heatmaps, which image-level AUROC does not capture.

These are single-run numbers on one category, without seed averaging or hyperparameter search, so
treat the ordering as indicative rather than a benchmark result.

## Repository structure

```
notebook.ipynb      Full implementation — all four methods, training and evaluation
notebook.pdf        Rendered notebook with all figures (readable without running anything)
notebook.tex        LaTeX export
notebook_files/     Figures: reconstructions, anomaly heatmaps, ROC curves
```

## Running it

The notebook was developed in Google Colab and expects a GPU.

```bash
pip install torch torchvision numpy matplotlib tqdm pillow scikit-learn
```

```bash
git clone https://github.com/M3nnoun/vision-anomaly-detection.git
cd vision-anomaly-detection
jupyter notebook notebook.ipynb
```

The first cells download and extract the MVTec AD Pill category; the rest run top to bottom.
If you only want to read the results, `notebook.pdf` has every figure already rendered.

## References

- Yang et al., *Unsupervised Anomaly Segmentation via Deep Feature Reconstruction* — [arXiv:2012.07122](https://arxiv.org/abs/2012.07122)
- Roth et al., *Towards Total Recall in Industrial Anomaly Detection* (PatchCore) — [arXiv:2106.08265](https://arxiv.org/abs/2106.08265) · [official implementation](https://github.com/amazon-science/patchcore-inspection)
- Bergmann et al., *MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection* (CVPR 2019)

## License

MIT
