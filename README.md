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
   conda env create -f dfdetection.yml

   

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
 Our multi-modal deepfake detection model significantly improves the accuracy of detecting fraudulent activities in financial systems by leveraging both visual and audio features. The use of a Vision Transformer for visual feature extraction and a CNN for audio feature extraction, combined with cross-modal attention and feature fusion, enables our model to detect inconsistencies that are often missed by single-modal methods. This approach not only enhances security but also provides a robust solution to combat financial fraud in the era of advanced deepfake technology
 
![Audio_visual_Deepfake](ModelArchitecure.jpg)
