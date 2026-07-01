# Tehran House Price Prediction using Linear Regression

## Overview
This project applies **Linear Regression** to predict **apartment prices in Tehran** using real estate listing data. The model was developed in Python using Scikit-learn and evaluated with common regression performance metrics.

This project was completed as part of my Machine Learning learning journey.

---

## Dataset
The dataset contains information about ~4,000 real apartments in Tehran, including:

- Area (m²)
- Number of Rooms
- Parking Availability
- Warehouse Availability
- Elevator Availability
- Address (Neighborhood in Tehran)
- Price in Toman (Target Variable)
- Price in USD

---

## Technologies Used
- Python
- Jupyter Notebook
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

---

## Project Workflow
1. Load and explore the dataset
2. Clean the data:
   - Remove rows with missing Address
   - Fix and convert Area column
   - Remove unrealistic outliers (Area > 700 m²)
   - Remove duplicate rows
3. Visualize relationships between features and price (EDA)
4. Engineer features:
   - Encode boolean columns to 0/1
   - Group rare neighborhoods into "Other"
   - Apply One-Hot Encoding on Address
   - Apply log transformation on Price
5. Split the dataset into training and testing sets
6. Train a Linear Regression model
7. Evaluate the model

---

## Features Used
- Area
- Room
- Parking
- Warehouse
- Elevator
- Address (encoded)

Target:
- Price (log-transformed, in Toman)

---

## Results

| Metric | Value |
|--------|-------|
| MAE    | 0.2731 |
| RMSE   | 0.4152 |
| R² Score | **0.8566** |

The model explains ~85% of the variation in apartment prices, showing that location (Address), area, and amenities are strong predictors of price in Tehran's real estate market.

---

## Repository Structure
```
├── House_Price_Predict.ipynb
├── House_Price_Predict.csv
└── README.md
```

---

## Future Improvements
- Random Forest Regression
- Gradient Boosting (XGBoost)
- Hyperparameter Tuning
- Cross Validation
- Interactive price prediction interface

---

## Author
**Diba Nazari**
- GitHub: https://github.com/dibanazari
- LinkedIn: *(Add your LinkedIn profile here)*

---

## License
This project is intended for educational and portfolio purposes.
