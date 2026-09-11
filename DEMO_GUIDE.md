# 🎓 SigVerify - Supervisor Demo Guide

## Overview
This demo showcases the completed work on SigVerify: Signature Verification with Explainable AI.

## Completed Features

### 1. Dataset Loader ✅
- Loads CEDAR dataset (50 genuine + 50 forged signatures)
- Preprocessing: resize to 128x128, normalize
- Creates training pairs for Siamese network

### 2. Siamese CNN Model ✅
- Shared weights architecture
- Conv2D layers: 32, 64, 128 filters
- Global Average Pooling
- 111,297 trainable parameters
- Achieves 100% validation accuracy

### 3. Grad-CAM Explainability ✅
- Visual heatmaps showing decision factors
- RED = strong influence on "Forged" decision
- GREEN = less influential areas
- Builds trust in AI decisions

## How to Run the Demo

### Quick Demo
```bash
python demo_supervisor.py