# ICS 485 – Machine Learning Term Project  
## Binary Classification with Active Learning Extension

This project focuses on solving a **binary classification problem** using a combination of traditional machine learning models and modern active learning techniques. The project is divided into two major parts:

- **Part A**: End-to-end supervised classification using multiple algorithms and comprehensive evaluation.
- **Part B**: Implementation of **active learning** strategies to reduce labeling cost while maintaining classification performance.

---

## Dataset

A labeled dataset was provided by the instructor for binary classification. In Part B, the same dataset is used **with most training labels hidden**, simulating a real-world labeling scenario for active learning.

---

## Part A – Binary Classification (70%)

### Dataset Analysis & Preprocessing
- Explored data structure, distributions, and correlations.
- Addressed missing values (if present) and performed cleaning.
- Handled class imbalance through oversampling/undersampling techniques.
- Performed feature engineering, selection, and transformation for optimal model input.

### Classifiers Implemented
We implemented **at least 4 different classifiers**, ensuring each of the following categories is covered:
- **Linear Model**: Logistic Regression or Support Vector Machine (SVM)
- **Tree-based or Distance-based**: Decision Trees or K-Nearest Neighbors (KNN)
- **Neural Network**: Feedforward network built using Keras or PyTorch
- **Ensemble Method**: Random Forest, AdaBoost, or similar

Each classifier includes:
- Proper **hyperparameter tuning** via cross-validation or a dedicated validation set.
- **Evaluation using multiple metrics**, with a focus on:
  - **G-Mean** (harmonic mean of Sensitivity and Specificity)
  - Precision, Recall, F1-score, ROC-AUC

### Evaluation and Error Analysis
- Presented a comparison of all models on the test set.
- Identified strengths and failure cases through confusion matrices and misclassification patterns.
- Suggested improvements (e.g., better feature encoding, tuning, ensemble stacking).

### Optional Enhancements Explored
- Additional classifiers for deeper comparison
- Dimensionality reduction (e.g., PCA)
- Margin-based analysis (SVM)
- Model interpretation using weights and rules
- Clustering-based labeling comparison

---

## Part B – Active Learning (30%)

In this part, we simulated a scenario where **only 50 training samples are initially labeled**. We gradually queried the model for labels using intelligent strategies to improve performance with **minimal labeling effort**.

### Strategies Implemented
- **Least Confidence Sampling** – selects samples the model is least confident about.
- **Entropy-Based Sampling** – selects samples with highest prediction entropy.

Each student was responsible for one strategy. The **logistic regression classifier** was used throughout to maintain consistency with Part A.

### Goal
Achieve results **comparable to Part A** with significantly fewer labeled instances, demonstrating the power of active learning in minimizing labeling costs.

---

## Tools & Libraries
- `scikit-learn` – Model training, evaluation, and preprocessing
- `matplotlib`, `seaborn` – Visualizations
- `pandas`, `numpy` – Data handling
- `Keras` / `PyTorch` – Neural network implementation (Part A)
- `Jupyter Notebooks` – All experiments documented in notebook format

---

## Course Information

- **Course**: ICS 485 – Machine Learning  
- **Institution**: King Fahd University of Petroleum and Minerals  
- **Term**: Second Semester, 2024–2025 (241)

## Licensing
This repository is developed as part of the KFUPM ICS 485 course (semester 241) and is intended solely for educational purposes.
