# Stutter-Detection
 
**Stutter Classification using Random Forest**

**Overview**

This repository contains code for a machine learning project focused on classifying different types of speech repetitions: prolongation, blocking, and normal speech. The project utilizes a Random Forest classifier trained on a curated dataset to accurately classify these speech patterns.

**Dataset**

The dataset used in this project comprises samples of speech instances categorized into three classes:
Repitition
Prolongation
Blocking
Normal speech


**Methodology**

**Feature Engineering**
Prior to training the model, the dataset underwent thorough feature engineering. The extracted features were normalized and processed to ensure optimal performance during training.

**Model Training**
A Random Forest classifier was employed for its ability to handle multi-class classification problems effectively. The classifier was trained using k-fold cross-validation to enhance model robustness and prevent overfitting.

**Results**

After training the Random Forest classifier with cross-validation, the model achieved an impressive accuracy of 93%. Evaluation metrics, including precision, recall, and F1-score, for each class are available in the project's notebook.
