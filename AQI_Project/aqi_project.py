# ===============================
# AQI PREDICTION PROJECT (FINAL)
# ===============================

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# -------------------------------
# 1. LOAD DATASET
# -------------------------------
df = pd.read_csv("aqi_dataset.csv")

print("\nDataset Preview:\n", df.head())

# -------------------------------
# 2. PREPROCESSING
# -------------------------------
df = df.dropna()

# Convert Date
df['Date'] = pd.to_datetime(df['Date'])
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day

df = df.drop('Date', axis=1)

# -------------------------------
# 3. FEATURES & TARGET
# -------------------------------
X = df[['PM2.5','PM10','NO2','SO2','CO','O3',
        'Temperature','Humidity','WindSpeed','Month','Day']]

y = df['AQI']

# -------------------------------
# 4. TRAIN-TEST SPLIT
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===============================
# 5. RANDOM FOREST MODEL
# ===============================
print("\n--- Random Forest Model ---")

rf_model = RandomForestRegressor(n_estimators=200, max_depth=10)
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

# Evaluation
mae_rf = mean_absolute_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print("Random Forest MAE:", mae_rf)
print("Random Forest R2 Score:", r2_rf)

# -------------------------------
# 6. LINEAR REGRESSION MODEL
# -------------------------------
print("\n--- Linear Regression Model ---")

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

y_pred_lr = lr_model.predict(X_test)

mae_lr = mean_absolute_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

print("Linear Regression MAE:", mae_lr)
print("Linear Regression R2 Score:", r2_lr)

# -------------------------------
# 7. FEATURE IMPORTANCE
# -------------------------------
print("\n--- Feature Importance (Random Forest) ---")

importance = rf_model.feature_importances_
features = X.columns

for i in range(len(features)):
    print(features[i], ":", round(importance[i], 4))

# -------------------------------
# 8. GRAPH (ACTUAL vs PREDICTED)
# -------------------------------
plt.scatter(y_test, y_pred_rf)
plt.xlabel("Actual AQI")
plt.ylabel("Predicted AQI")
plt.title("Actual vs Predicted AQI (Random Forest)")
plt.savefig("aqi_graph.png")
print("Graph saved as aqi_graph.png")

# -------------------------------
# 9. FINAL MESSAGE
# -------------------------------
print("\nModel training and evaluation completed successfully!")