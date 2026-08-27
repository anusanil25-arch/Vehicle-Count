# Vehicle-Count
#  Vehicle Count Prediction from Sensor Data

## 📌 Project Overview

This project predicts the number of vehicles recorded by traffic sensors using historical vehicle counts and time-based features.

### Problem Type
Supervised Machine Learning - Regression

### Target Variable
`Vehicles`

### Dataset
`vehicles.csv`

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn


##  Project Workflow

1. Load the dataset
2. Data inspection
3. Data cleaning
4. DateTime conversion
5. Sensor block identification
6. Feature engineering
7. Lag feature creation
8. Rolling mean calculation
9. Exploratory Data Analysis
10. Train-test split
11. Model training
12. Model evaluation
13. Model comparison
14. Best model selection

##  Feature Engineering

The project creates time-based features such as:

- Year
- Month
- Day
- Hour
- Day of Week
- Day of Year
- Week of Year
- Is Weekend

Lag features:

- Lag 1
- Lag 2
- Lag 3
- Lag 24


Rolling mean features:

- Rolling Mean 3
- Rolling Mean 24


##  Machine Learning Models

Three regression models were compared:

1. Linear Regression
2. Random Forest Regressor
3. Gradient Boosting Regressor

##  Evaluation Metrics

The models were evaluated using:

- MAE
- RMSE
- R² Score

Lower MAE and RMSE indicate better performance, while a higher R² indicates better performance.

##  Result

According to the benchmark in the project notebook, Hist Gradient Boosting performed best:

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Linear Regression | 19.895710 | 250426583| 0.133760 |
| Random Forest | 18.835012 | 24.450652 | 0.198980 |
| Hist Gradient Boosting | 18.192314 | 24.616311 | 0.188089 |



##  Visualizations

The project includes:

- Vehicle count distribution
- Average traffic by hour
- Average traffic by day of week
- Average traffic by month
- Vehicle counts over time
- Correlation matrix
- Actual vs Predicted vehicle counts


##  Project Files

- `Vehicle_Count_Prediction_Project.ipynb` — Main Jupyter Notebook
- `vehicles.csv` — Dataset


##  Conclusion

The project demonstrates how historical vehicle counts and time-based features can be used to predict future vehicle traffic using regression algorithms.

 indicates that Random Forest provides the best performance among the three tested models.

##  Author

**Anu S Anil**




