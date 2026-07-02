# 🩺 Drug Prescription Classifier

A machine learning project that predicts the most appropriate drug for a patient based on their medical attributes, using a **Decision Tree Classifier**.

## 📊 Dataset

The dataset (`drug200.csv`) contains 200 patient records with the following features:

| Column | Description |
|---|---|
| `Age` | Patient's age |
| `Sex` | Patient's sex (M/F) |
| `BP` | Blood pressure level (LOW / NORMAL / HIGH) |
| `Cholesterol` | Cholesterol level (NORMAL / HIGH) |
| `Na_to_K` | Sodium-to-potassium ratio in blood |
| `Drug` | Target label — the prescribed drug (drugA, drugB, drugC, drugX, drugY) |

## 🧠 Approach

1. **Load & Explore Data** — read the CSV and inspect its structure.
2. **Preprocessing** — encode categorical columns (`Sex`, `BP`, `Cholesterol`) into numeric values using `LabelEncoder`.
3. **Train/Test Split** — 70% of the data used for training, 30% for testing.
4. **Model** — a `DecisionTreeClassifier` with `criterion="entropy"` and `max_depth=4`.
5. **Evaluation** — accuracy measured on the test set.
6. **Visualization** — the trained tree is visualized using `sklearn.tree.plot_tree`.

## ✅ Results

**Accuracy: 98.3%**

The `Na_to_K` feature had the strongest influence on the model's decisions.

## 🛠️ Tech Stack

- Python
- pandas
- numpy
- scikit-learn
- matplotlib

## 🚀 Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/<repo-name>.git
   cd <repo-name>
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Open the notebook:
   ```bash
   jupyter notebook DrugPrescription.ipynb
   ```

## 📁 Project Structure

```
.
├── DrugPrescription.ipynb   # Main notebook
├── drug200.csv               # Dataset
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

## 📌 Conclusion

The Decision Tree model achieved **98.3% accuracy** in predicting the appropriate drug based on patient features, showing that a relatively simple, interpretable model can perform very well on this kind of structured medical data.
