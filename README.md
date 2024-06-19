# Concept-Detection
 Code for the participation of team SSNMLRGKSR at the ImageCLEFmedical Caption task of 2024. 

 
**Input**

ImageCLEF Concept Detection 2024 dataset from **https://www.imageclef.org/2024/medical/caption**

**Feature Extraction**

Features extracted based on DenseNet is saved under the file named **“mlb_dp_classifier.pkl”**
Features extracted based on MultiLabelBinazier (MLB) is saved under the file named** “mlb_bpo_classifier.pkl”**

**Model Generation**

Model generated based on DenseNet is saved under the file named **“dp-classifier.h5”**
Model generated based on MLB is saved under the file named** “bpo-classifier.h5”**

**Output**

**DenseNet output - ** dp-only-dp-labels.csv
**MLB output  -** bpo-only-bpo-labels.csv
**DenseNet with MLB output - ** bpo-only-bpo-labels.csv

**Code**

Concept detection.py

**How to create environment, develop concept detection model and detect concepts?**

**Environment creation**

conda create --name <my-env>

**Activate environment**

conda activate <my-env>

**Install required packages** as given in “requirement.txt”

**To develop concept detection model and detect concepts**

**Run**

python3 Concept detection.py

// This program takes the medical images and respective concepts from training set as input to generate models based on DenseNet and MultiLabelBinazier.

// Intermediate results – Two feature vector files in .pkl format  and two model file in .h5 format.

//Output – One or more concepts for each images in the test set.



