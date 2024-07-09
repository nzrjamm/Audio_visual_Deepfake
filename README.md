# Audio_visual_Deepfake

# Deepfake Detection Project

## Introduction

This repository contains code for detecting deepfakes using audio-visual modalities and Vision Transformer (DeiT).

## Setup

### Environment

1. **Create Conda Environment:**
   ```bash
   conda create --name dfdetection python=3.8
   conda activate dfdetection
   pip install -r requirements.txt  # or use dfdetection.yml

### Dataset Structure 
dataset/
├── train/
│   ├── real/
│   └── fake/
├── val/
│   ├── real/
│   └── fake/
└── test/
    ├── real/
    └── fake/
###Prepocessing 
scrtipts of preprocesing is in the src folder.
src/
 'train_preprocessing/', 'val_preprocessing/', and 'test_preprocessing/'.

 ### Training
 Run ptyhon trianing.py file to train the model.
 ## Do the same the testing and val .py files as well.

 ### Model Archirecture 
 
![Audio_visual_Deepfake](ModelArchitecure.jpg)
