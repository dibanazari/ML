# 🔬 Breast Cancer Cell Classification using SVM

A machine learning project that classifies cell samples as **Benign** or **Malignant** using a Support Vector Machine (SVM) with an RBF kernel.

## 📋 Overview

This project builds an end-to-end machine learning pipeline to predict whether a breast cell sample is benign or malignant, based on a set of cell characteristics (clump thickness, uniformity of cell size/shape, etc.). The workflow covers data exploration, preprocessing, model training, and evaluation.

## 🛠️ Tech Stack

- Python 3
- pandas / NumPy — data handling
- scikit-learn — modeling and evaluation
- matplotlib — visualization

## 📂 Dataset

The dataset (`cell_samples.csv`) contains the following features for each cell sample:

| Feature | Description |
|---|---|
| Clump | Clump Thickness |
| UnifSize | Uniformity of Cell Size |
| UnifShape | Uniformity of Cell Shape |
| MargAdh | Marginal Adhesion |
| SingEpiSize | Single Epithelial Cell Size |
| BareNuc | Bare Nuclei |
| BlandChrom | Bland Chromatin |
| NormNucl | Normal Nucleoli |
| Mit | Mitoses |
| Class | Target label (2 = Benign, 4 = Malignant) |

## 🔄 Workflow

1. **Setup & Load Data** — Import libraries and load the dataset.
2. **Exploratory Data Analysis** — Visualize class separability using scatter plots.
3. **Data Preprocessing** — Clean invalid values and prepare feature/target arrays.
4. **Train/Test Split** — Split data into 80% training and 20% test sets.
5. **Model Training** — Train an SVM classifier with an RBF kernel.
6. **Model Evaluation** — Assess performance with a Confusion Matrix, classification report, F1-score, and Jaccard index.

## 📊 Results

| Metric | Score |
|---|---|
| F1-score (weighted) | ~0.96 |
| Jaccard Score | ~0.94 |

The model achieves strong performance in distinguishing malignant from benign cell samples.

## 🚀 How to Run

```bash
pip install pandas numpy scikit-learn matplotlib
jupyter notebook Untitled13.ipynb
```

## 📁 Project Structure

```
├── Untitled13.ipynb      # Main notebook with full pipeline
├── cell_samples.csv      # Dataset
└── README.md             # Project documentation
```

## 📄 License

This project is open source and available for educational purposes.
