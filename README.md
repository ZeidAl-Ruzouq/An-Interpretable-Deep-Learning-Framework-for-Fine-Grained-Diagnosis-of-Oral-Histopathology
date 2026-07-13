# Reproducible Benchmarks for Interpretable Deep Learning in Oral Histopathology

This repository contains the code and experimental pipelines used for three complementary benchmarks in our study on interpretable deep learning for fine-grained diagnosis of oral histopathology.

The work is organized into three main experimental components:

- Bench 1: Initial model training and evaluation
- Bench 2: Cost-aware graph neural network ensemble (CA-GNN-E)
- Bench 3: Generalization testing on an external cohort (CPTAC-HNSCC)

## Overview

The framework combines several state-of-the-art deep learning backbones, including ConvNeXt, ResNet-50, EfficientNet-B0, DeiT Small, and RegNet-Y 8GF, with explainability analysis and ensemble-based decision fusion. The implementation is designed to support reproducible experimentation, model evaluation, and external validation for oral histopathology classification.

---

## BENCH 1 — Initial Models Training

### Purpose
This benchmark focuses on training and evaluating baseline deep learning models for oral histopathology classification.

### Main scripts
- `Initial Models Training   ---   BENCH 1/orchid_pipeline.py`
  - Main training and evaluation pipeline
  - Supports multiple architectures and explainability methods
- `Initial Models Training   ---   BENCH 1/preparedata.py`
  - Dataset preparation utility for downloading and organizing the ORCHID dataset

---

## BENCH 2 — CA-GNN-E Model

### Purpose
This benchmark introduces a cost-aware graph neural network ensemble that combines predictions from multiple base models and evaluates the trade-off between accuracy and computational cost.

### Main script
- `CA-GNN-E Model  ---  BENCH 2/CA-GNN-E_pipeline.py`
  - Builds graph-based ensemble predictions from base-model outputs
  - Uses cost-aware evaluation and Optuna-based hyperparameter tuning

---

## BENCH 3 — Generalization Testing

### Purpose
This benchmark evaluates the robustness and generalization capability of the trained models and ensemble on an external dataset, specifically the CPTAC-HNSCC cohort.

### Main script
- `Generalization Testing --- BENCH 3/CPTAC-HNSCC/TCGA_Test.py`
  - Runs inference and evaluation on external whole-slide image data
  - Supports base-model inference and CA-GNN-E evaluation

---

## Demo Website

A live demo website for the project is available here:

https://oralai-72ye.onrender.com/

The website is intended to be publicly accessible, but if you would like to test it out or request access for demonstration purposes, please contact:

zalruzouq@gmail.com

---

## Notes for Reproducibility

- Detailed setup instructions for each benchmark are available in the corresponding `How to run.md` files.
- The scripts in this repository are intended for research and reproducibility purposes.
