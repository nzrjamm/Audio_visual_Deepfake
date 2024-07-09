# Audio_visual_Deepfake

## Introduction

This repository contains code for detecting deepfakes using audio-visual modalities and Vision Transformer (DeiT).

## Setup

### Environment

1. **Create Conda Environment:**
   ```bash
   conda create --name dfdetection python=3.12
   conda activate dfdetection
   pip install -r requirements.txt  # or use dfdetection.yml

### Dataset Structure 
dataset/

└── train/
    ├── real/ 
    └── fake/
    
└── Val/
    ├── real/  
    └── fake/

└── test/
    ├── real/  
    └── fake/
- **Dataset Link:** The `[DFDC (DeepFake Detection Challenge) dataset](https://github.com/ondyari/FaceForensics/tree/master/datase](https://www.kaggle.com/c/deepfake-detection-challenge)` 
    
###Prepocessing 
scrtipts of preprocesing is in the src folder.
src/
 'train_preprocessing/', 'val_preprocessing/', and 'test_preprocessing/'.

 ### Training
 Run ptyhon trianing.py file to train the model.
 Do the same the testing and val .py files as well.

 ### Model Archirecture 
 
![Audio_visual_Deepfake](ModelArchitecure.jpg)
