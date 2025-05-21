# Advanced Skin Lesion Classification using Deep Learning

![Skin Lesion Classification Banner](https://via.placeholder.com/1200x300?text=Skin+Lesion+Classification)

## 📌 Project Overview

This project implements a state-of-the-art deep learning system for the multi-class classification of skin lesions using the ISIC 2019 dataset. Using enhanced transfer learning with InceptionV3 architecture, combined with attention mechanisms, residual blocks, and focal loss, our model achieves high accuracy in classifying dermatological conditions.

### Key Features

- **Multi-class classification** of 9 skin lesion types
- **Enhanced InceptionV3 architecture** with attention mechanisms and residual blocks
- **Focal loss** implementation for handling class imbalance
- **Two-stage training** with progressive fine-tuning
- **Explainable AI** using LIME for interpretable predictions
- **Optimized hyperparameters** using Grey Wolf Optimizer

## 🔍 Dataset

This project uses the ISIC 2019 Challenge dataset, which contains over 25,000 dermoscopic images across 9 diagnostic categories:
- Melanoma (MEL)
- Melanocytic nevus (NV)
- Basal cell carcinoma (BCC)
- Actinic keratosis (AK)
- Benign keratosis (BKL)
- Dermatofibroma (DF)
- Vascular lesion (VASC)
- Squamous cell carcinoma (SCC)
- None of the above (UNK)

### Downloading the Dataset

1. Visit [ISIC Challenge 2019](https://challenge.isic-archive.com/landing/2019/)
2. Download:
   - Training images (`ISIC_2019_Training_Input.zip`)
   - Ground truth (`ISIC_2019_Training_GroundTruth.csv`)
3. Extract and organize: