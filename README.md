# Vehicle Count Prediction Using Machine Learning

## 📌 Project Overview

This project focuses on predicting the number of vehicles using historical traffic sensor data and Machine Learning techniques.

Traffic data changes based on time and historical traffic conditions. By analyzing previous vehicle counts and extracting useful features from date and time information, a Machine Learning model can learn traffic patterns and predict future vehicle counts.

This project includes the complete Machine Learning workflow:

* Data loading
* Data understanding
* Data cleaning
* Data preprocessing
* Feature engineering
* Exploratory Data Analysis (EDA)
* Data visualization
* Model training
* Model evaluation
* Model comparison
* Best model selection
* Vehicle count prediction

---

# Project Objective

The main objective of this project is to build a Machine Learning model that can predict the number of vehicles at a particular date and time using historical traffic data.

The project aims to:

* Understand the vehicle sensor dataset.
* Clean and preprocess the data.
* Extract useful time-based features from the `DateTime` column.
* Create historical traffic features using lag values.
* Create rolling mean features to capture recent traffic trends.
* Perform Exploratory Data Analysis.
* Train multiple regression models.
* Compare the performance of different models.
* Select the best-performing model.
* Predict the vehicle count for a specific timestamp.

---

# Machine Learning Problem Type

This project is a **Supervised Machine Learning Regression Problem**.

### Why Regression?

The target variable is `Vehicles`, which contains numerical values representing the number of vehicles.

Regression algorithms are used when the goal is to predict a numerical value.

---

#  Dataset Description

The original dataset contains:

* **48,120 rows**
* **2 columns**

| Column     | Description                                      |
| ---------- | ------------------------------------------------ |
| `DateTime` | Date and time when the vehicle data was recorded |
| `Vehicles` | Number of vehicles recorded at that timestamp    |

---

#  Data Understanding

The following methods were used to understand the dataset:

```python
df.head()
```

Used to display the first few rows of the dataset.

```python
df.shape
```

Used to check the number of rows and columns.

```python
df.info()
```

Used to understand:

* Column names
* Data types
* Non-null values
* Memory usage

```python
df.describe()
```

Used to obtain statistical information about numerical columns.

```python
df.columns
```

Used to check the column names.

---

#  Data Cleaning and Preprocessing

## 1. Checking Missing Values

Missing values were checked using:

```python
df.isnull().sum()
```

The original dataset not contain missing values.

---

## 2. Checking Duplicate Records

Duplicate rows were checked using:

```python
df.duplicated().sum()
```

A total of **1,082 duplicate records** were identified.

The duplicate rows were removed using:

```python
df = df.drop_duplicates()
```

After removing duplicates, the dataset contained:

* **47,038 rows**

---

## 3. Converting DateTime

The `DateTime` column was initially stored as an object/string type.

It was converted into datetime format using:

```python
df['DateTime'] = pd.to_datetime(df['DateTime'])
```

This conversion allows useful date and time information to be extracted.

---

## 4. Sorting Data by Time

Since this is time-based traffic data, the dataset was sorted chronologically.

```python
df = df.sort_values('DateTime')
df = df.reset_index(drop=True)
```


---

#  Feature Engineering

Feature engineering is the process of creating useful input features from the existing data.

Several time-based features were extracted from the `DateTime` column.

## Time-Based Features

### Date

```python
df['date'] = df['DateTime'].dt.day
```

Represents the day of the month.

---

### Weekday

```python
df['weekday'] = df['DateTime'].dt.weekday
```

Represents the day of the week.

---

### Hour

```python
df['hour'] = df['DateTime'].dt.hour
```

Represents the hour when the vehicle data was recorded.

---

### Month

```python
df['month'] = df['DateTime'].dt.month
```

Represents the month.

---

### Year

```python
df['year'] = df['DateTime'].dt.year
```

Represents the year.

---

### Day of Year

```python
df['dayofyear'] = df['DateTime'].dt.dayofyear
```

Represents the numerical day within the year.

---

### Week of Year

```python
df['weekofyear'] = (
    df['DateTime']
    .dt.isocalendar()
    .week
    .astype(int)
)
```

Represents the week number within the year.

---

# Lag Features

Lag features represent previous vehicle observations.

The following lag features were created:

```python
df['Lag_1'] = df['Vehicles'].shift(1)

df['Lag_2'] = df['Vehicles'].shift(2)

df['Lag_3'] = df['Vehicles'].shift(3)

df['Lag_24'] = df['Vehicles'].shift(24)
```

## Lag Feature Description

| Feature  | Description                                          |
| -------- | ---------------------------------------------------- |
| `Lag_1`  | Vehicle count from the previous observation          |
| `Lag_2`  | Vehicle count from two previous observations         |
| `Lag_3`  | Vehicle count from three previous observations       |
| `Lag_24` | Vehicle count from twenty-four previous observations |

Lag features help the model learn from historical traffic patterns.

---

#  Rolling Mean Features

Rolling mean features calculate the average vehicle count over previous observations.

The following features were created:

```python
df['Rolling_Mean_3'] = df['Vehicles'].shift(1).rolling(3).mean()

df['Rolling_Mean_24'] = df['Vehicles'].shift(1).rolling(24).mean()
```

## Rolling Mean Features

| Feature           | Description                             |
| ----------------- | --------------------------------------- |
| `Rolling_Mean_3`  | Average of the previous 3 observations  |
| `Rolling_Mean_24` | Average of the previous 24 observations |


---

# Handling Missing Values After Feature Engineering

Lag and rolling mean features create missing values at the beginning of the dataset.


These rows were removed using:

```python
df = df.dropna().reset_index(drop=True)
```

---

#  Exploratory Data Analysis

Exploratory Data Analysis was performed to understand traffic patterns and relationships in the dataset.

---

## 1. Vehicle Count Distribution

A histogram was used to analyze the distribution of vehicle counts.

```python
plt.figure(figsize=(10,5))

sns.histplot(df['Vehicles'], bins=50)

plt.title('Distribution of Vehicle Counts')

plt.xlabel('Number of Vehicles')

plt.ylabel('Frequency')

plt.show()
```

This visualization helps understand the frequency and variation of vehicle counts.

---

## 2. Average Vehicle Count by Hour

```python
hourly = df.groupby('hour')['Vehicles'].mean()

plt.figure(figsize=(10,5))

hourly.plot()

plt.title('Average Vehicle Count by Hour')

plt.xlabel('Hour')

plt.ylabel('Average Vehicles')

plt.grid()

plt.show()
```

This analysis helps understand how traffic changes throughout the day.

---

## 3. Average Vehicle Count by Day of Week

```python
weekly = df.groupby('weekday')['Vehicles'].mean()

plt.figure(figsize=(10,5))

weekly.plot(kind='bar')

plt.title('Average Vehicle Count by Day of Week')

plt.xlabel('Day of Week')

plt.ylabel('Average Vehicles')

plt.show()
```

This helps analyze weekly traffic patterns.

---

## 4. Average Vehicle Count by Month

```python
monthly = df.groupby('month')['Vehicles'].mean()

plt.figure(figsize=(10,5))

monthly.plot(kind='bar')

plt.title('Average Vehicle Count by Month')

plt.xlabel('Month')

plt.ylabel('Average Vehicles')

plt.show()
```

This helps understand monthly traffic patterns.

---

## 5. Correlation Matrix

A correlation heatmap was created to study the relationship between numerical features.

```python
plt.figure(figsize=(12,8))

sns.heatmap(
    df.corr(numeric_only=True),
    annot=True,
    cmap='coolwarm'
)

plt.title('Correlation Matrix')

plt.show()
```

The correlation matrix helps understand the relationship between input features and the target variable.

---

# Feature and Target Selection

The target variable is:

```python
y = df['Vehicles']
```

The input features are:

```python
x = df.drop(
    ['DateTime', 'Vehicles'],
    axis=1
)
```

## Input Features

The final input features include:

```text
date
weekday
hour
month
year
dayofyear
weekofyear
Lag_1
Lag_2
Lag_3
Lag_24
Rolling_Mean_3
Rolling_Mean_24
```

## Target

```text
Vehicles
```

---

#  Train-Test Split

The dataset was split chronologically.

```python
split = int(len(df) * 0.8)

x_train = x.iloc[:split]

x_test = x.iloc[split:]

y_train = y.iloc[:split]

y_test = y.iloc[split:]
```

## Split Ratio

* **80% Training Data**
* **20% Testing Data**

The data was not randomly shuffled because this is time-based data.

The model is trained using earlier observations and tested using later observations.

---

#  Machine Learning Models

The following regression models were used.

---

## 1. Linear Regression

Linear Regression was used as a baseline model.

```python
lr = LinearRegression()

lr.fit(x_train, y_train)

y_pred_lr = lr.predict(x_test)
```

Linear Regression attempts to find a linear relationship between input features and the target variable.

---

## 2. Random Forest Regressor

Random Forest is an ensemble learning algorithm that combines predictions from multiple decision trees.

```python
rf = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

rf.fit(x_train, y_train)

y_pred_rf = rf.predict(x_test)
```

### Parameters

* `n_estimators=100` → Uses 100 decision trees.
* `random_state=42` → Ensures reproducible results.
* `n_jobs=-1` → Uses available CPU cores.

---

## 3. Gradient Boosting Regressor

Gradient Boosting builds models sequentially and attempts to improve the errors made by previous models.

The model was trained and evaluated using the same training and testing data.

---

#  Model Evaluation Metrics

Three metrics were used to evaluate the models.

---

## MAE – Mean Absolute Error

MAE measures the average absolute difference between actual and predicted values.

**Lower MAE is better.**

---

## RMSE – Root Mean Squared Error

RMSE measures prediction error and gives more importance to larger errors.

**Lower RMSE is better.**

---

## R² Score

R² Score measures how much variation in the target variable is explained by the model.

**Higher R² Score is generally better.**

---

#  Model Performance Comparison

The results obtained from the project are:

| Model             |           MAE |          RMSE |     R² Score |
| ----------------- | ------------: | ------------: | -----------: |
| Linear Regression |     19.895710 |     25.426583 |     0.133760 |
| Random Forest     |     18.835012 | **24.450652** | **0.198980** |
| Gradient Boosting | **18.192314** |     24.616311 |     0.188089 |

---

#  Best Model

## Random Forest Regressor

Random Forest was selected as the best overall model.

### Best Results

```text
RMSE: 24.450652

R² Score: 0.198980
```

Random Forest achieved:

* The **lowest RMSE**
* The **highest R² Score**

Although Gradient Boosting achieved a slightly lower MAE, Random Forest was selected based on its overall performance.

---

#  Final Model

The final Random Forest model was created using:

```python
best_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

best_model.fit(x_train, y_train)
```

---

# Actual vs Predicted Vehicle Count

The final model was used to generate predictions on the test dataset.

```python
y_pred_final = best_model.predict(x_test)
```

Actual and predicted values were compared using:

```python
comparison = pd.DataFrame({
    'Actual Vehicles': y_test.values,
    'Predicted Vehicles': y_pred_final
})

comparison.head(10)
```

A graph was also created to compare the first 100 actual and predicted values.

```python
plt.figure(figsize=(12, 5))

plt.plot(
    y_test.values[:100],
    label='Actual'
)

plt.plot(
    y_pred_final[:100],
    label='Predicted'
)

plt.xlabel('Test Samples')

plt.ylabel('Vehicle Count')

plt.title('Actual vs Predicted Vehicle Count')

plt.legend()

plt.show()
```

---

#  Vehicle Count Prediction

The final model was used to predict vehicle count for the following timestamp:

```text
2017-06-30 18:00:00
```

Historical data before the prediction time was used to create all required features.

The following information was created for prediction:

* Date
* Weekday
* Hour
* Month
* Year
* Day of Year
* Week of Year
* Lag_1
* Lag_2
* Lag_3
* Lag_24
* Rolling_Mean_3
* Rolling_Mean_24

The model prediction was generated using:

```python
prediction = best_model.predict(input_data)
```

## Final Prediction Result

```text
Prediction Date & Time:
2017-06-30 18:00:00

Predicted Vehicle Count:
42
```

#  Predicted Vehicle Count: 42

---

# 🔄 Project Workflow

```text
Problem Understanding
        ↓
Data Loading
        ↓
Data Understanding
        ↓
Data Cleaning
        ↓
DateTime Conversion
        ↓
Feature Engineering
        ↓
Lag Features
        ↓
Rolling Mean Features
        ↓
Exploratory Data Analysis
        ↓
Feature and Target Selection
        ↓
Chronological Train-Test Split
        ↓
Model Training
        ↓
Model Evaluation
        ↓
Model Comparison
        ↓
Best Model Selection
        ↓
Final Prediction
```

---

#  Technologies Used

## Programming Language

* Python

## Libraries

* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn

---


---



---

#  Suggested Project Structure

```text


├── vehicle_count.ipynb
│
├── vehicles.csv
│
├── README.md

```

---



#  Key Learnings

Through this project, I learned:

* Data preprocessing techniques
* Handling duplicate data
* Working with DateTime data
* Feature engineering
* Creating lag features
* Creating rolling mean features
* Exploratory Data Analysis
* Data visualization
* Regression algorithms
* Random Forest
* Gradient Boosting
* Model evaluation
* MAE, RMSE, and R² Score
* Model comparison
* Time-based train-test splitting
* Making predictions using trained Machine Learning models

---

#  Limitations

The model has some limitations:

* It depends on the features available in the dataset.
* Additional real-world factors that affect traffic are not included.
* The model performance can potentially be improved through further feature engineering and hyperparameter tuning.
* The current prediction approach depends on historical vehicle observations.

---

#  Future Improvements

This project can be improved by:

* Adding more traffic-related data.
* Performing hyperparameter tuning.
* Using advanced time-series forecasting models.
* Adding external factors that may affect traffic.
* Developing a real-time traffic prediction system.
* Deploying the model as a web application.

---

# 📌 Conclusion

This project demonstrates how Machine Learning can be used to predict vehicle counts using historical sensor data.

The project began with a dataset containing `DateTime` and `Vehicles` columns. The data was cleaned, duplicate records were removed, and the DateTime column was converted into a usable datetime format.

Several time-based features were extracted, including hour, weekday, month, year, day of year, and week of year. Historical traffic features such as lag values and rolling averages were also created.

Three regression models were trained:

* Linear Regression
* Random Forest Regressor
* Gradient Boosting Regressor

The models were evaluated using MAE, RMSE, and R² Score.

Among the tested models, **Random Forest Regressor achieved the best overall performance**, with the lowest RMSE and highest R² Score.

Finally, the trained model was used to predict the vehicle count for a specific timestamp.

### Final Prediction

**Prediction Date & Time:** `2017-06-30 18:00:00`

**Predicted Vehicle Count:** `42`

This project shows how historical traffic data and Machine Learning techniques can be combined to support vehicle count prediction.

---

# Author

**Anu**

---


