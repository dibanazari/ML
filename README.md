# ❤️ Heart Disease Risk Prediction using SVM

A machine learning project that predicts a patient's heart disease risk (**High / Low**) using a **Support Vector Machine (SVM)**, tuned with `GridSearchCV`.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![scikit--learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 Overview

This project walks through a complete supervised ML workflow — from raw data to a tuned, evaluated classifier:

`EDA → Preprocessing → Model Training (SVM + GridSearchCV) → Evaluation`

The goal is to classify patients as **high risk** or **low risk** of heart disease based on 13 clinical features (age, chest pain type, cholesterol, max heart rate, etc.).

## 📊 Dataset

- **Source:** [Heart Attack Analysis & Prediction Dataset](https://www.kaggle.com/datasets/rashikrahmanpritom/heart-attack-analysis-prediction-dataset) (Kaggle) — *update this link if your source is different*
- **Rows:** 303 patient records
- **Target:** `output` (1 = high risk, 0 = low risk)

| Column | Description |
|---|---|
| `age` | Age in years |
| `sex` | Sex (1 = male, 0 = female) |
| `cp` | Chest pain type (0–3) |
| `trtbps` | Resting blood pressure (mm Hg) |
| `chol` | Serum cholesterol (mg/dl) |
| `fbs` | Fasting blood sugar > 120 mg/dl (1 = true, 0 = false) |
| `restecg` | Resting ECG results (0–2) |
| `thalachh` | Maximum heart rate achieved |
| `exng` | Exercise-induced angina (1 = yes, 0 = no) |
| `oldpeak` | ST depression induced by exercise |
| `slp` | Slope of the peak exercise ST segment |
| `caa` | Number of major vessels colored by fluoroscopy (0–4) |
| `thall` | Thalassemia type |
| `output` | Target: 1 = high risk, 0 = low risk |

> ⚠️ Place `heart.csv` in the project root before running the notebook (not included in this repo — download it from the Kaggle link above).

## 🧠 Methodology

1. **EDA** — feature distributions and the relationship between chest pain type (`cp`) and max heart rate (`thalachh`) across risk groups.
2. **Preprocessing** — 80/20 stratified train/test split, feature scaling with `StandardScaler`.
3. **Model training** — `SVC` tuned via `GridSearchCV` (5-fold CV) over `C`, `gamma`, and `kernel`.
4. **Evaluation** — accuracy, confusion matrix, precision, recall, and F1-score on the held-out test set.

## 📈 Results

| Metric | Score |
|---|---|
| Best CV Accuracy | 0.822 |
| Test Accuracy | **0.852** |
| Precision | 0.833 |
| Recall | 0.909 |
| F1-score (weighted) | 0.851 |

**Best hyperparameters:** `{'C': 1, 'gamma': 0.1, 'kernel': 'rbf'}`

## 🗂️ Project Structure

```
.
├── heart_disease_risk_svm.ipynb   # Main notebook (EDA → model → evaluation)
├── heart.csv                       # Dataset (download separately, see above)
├── requirements.txt
├── README.md
└── LICENSE
```

## 🚀 Getting Started

```bash
# Clone the repo
git clone https://github.com/<your-username>/heart-disease-risk-svm.git
cd heart-disease-risk-svm

# Install dependencies
pip install -r requirements.txt

# Launch the notebook
jupyter notebook heart_disease_risk_svm.ipynb
```

## 🛠️ Tech Stack

Python · pandas · NumPy · scikit-learn · matplotlib

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
