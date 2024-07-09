# Audio_visual_Deepfake

## Introduction

This repository contains code for detecting deepfakes using audio-visual modalities and Vision Transformer (DeiT).

## Setup

### Environment

1. **Create Conda Environment:**
   ```bash
   conda create --name dfdetection
   conda activate dfdetection
   pip install -r requirements.txt  # or use dfdetection.yml
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


The DFDC dataset is available on Kaggle. You can download it [here](https://www.kaggle.com/c/deepfake-detection-challenge).
    
###Prepocessing 
scrtipts of preprocesing is in the src folder.
src/
 'train_preprocessing/', 'val_preprocessing/', and 'test_preprocessing/'.

 ### Training
 Run ptyhon trianing.py file to train the model.
 Do the same the testing and val .py files as well.

 ### Model Archirecture 
 
![Audio_visual_Deepfake](ModelArchitecure.jpg)
