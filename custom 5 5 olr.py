# Name:Ahnaf Irfan
# Task: Task 5.5(Ordinary Linear Regression) 
# Description: This script trains and evaluates an Ordinary Linear Regression model
#              on the Boston housing dataset (from ISLP).I did a mean squared error for the evaluation of the prediction model. 
#               It includes calculation of RSS, residual
#              statistics, and residual plots for both training and test sets.



# Step 1: Import libraries
import matplotlib.pyplot as plt
import pandas as pd
from ISLP import load_data
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Step 2: Loaded the Boston dataset
boston = load_data("Boston")
X = boston.drop(columns='medv')  # Features
y = boston['medv']               # Target (Median value of owner-occupied homes)

# Step 3: Split data into training and test sets evenly(50/50)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Step 4: Trained the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Predicted on the test data
y_pred = model.predict(X_test)

# Step 6: Evaluated the model
mse = mean_squared_error(y_test, y_pred)

# Computing residuals with train data
y_train_pred = model.predict(X_train)
train_residuals = y_train - y_train_pred

# RSS on training data
rss_train = np.sum(train_residuals ** 2)

# Computing residuals with test data
y_test_pred = model.predict(X_test)
test_residuals = y_test - y_test_pred

# RSS on test data
rss_test = np.sum(test_residuals ** 2)

#--- RESIDUAL STATISTICS (on train data) ---
max_residual_train = np.max(np.abs(train_residuals))
mean_abs_residual_train = mean_absolute_error(y_train, y_train_pred)
mean_residual_train = np.mean(train_residuals)

# --- RESIDUAL STATISTICS (on test data) ---
max_residual_test = np.max(np.abs(test_residuals))
mean_abs_residual_test = mean_absolute_error(y_test, y_test_pred)
mean_residual_test = np.mean(test_residuals)

# Step 7: Evaluation between predicted and test data
print("\nMean Squared Error (MSE):", mse)

#  OUTPUT for RSS
print(f"RSS (Train): {rss_train:.2f}")
print(f"RSS (Test): {rss_test:.2f}")

#Output for RSS's statistics(train)
print("\nResidual Statistics (Test Data):")
print(f"  Max Residual (Absolute): {max_residual_train:.2f}")
print(f"  Mean Absolute Residual: {mean_abs_residual_train:.2f}")
print(f"  Mean Residual: {mean_residual_train:.2f}")

#Output for RSS's statistics(test)
print(f"  Max Residual (Absolute): {max_residual_test:.2f}")
print(f"  Mean Absolute Residual: {mean_abs_residual_test:.2f}")
print(f"  Mean Residual: {mean_residual_test:.2f}")

# Plot
plt.figure(figsize=(10, 5))

# Plotting of train data
plt.subplot(1, 2, 1)
plt.scatter(y_train, train_residuals, alpha=0.7)
plt.axhline(y=0, color='black', linestyle='--')
plt.title("Train Data Residual Plot")
plt.xlabel("Actual MEDV") #prediction goal
plt.ylabel("Residuals")

# Plotting of test data
plt.subplot(1, 2, 2)
plt.scatter(y_test, test_residuals, alpha=0.7, color='orange')
plt.axhline(y=0, color='black', linestyle='--')
plt.title("Test Data Residual Plot")
plt.xlabel("Actual MEDV") #prediction_goal
plt.ylabel("Residuals")

plt.tight_layout()
plt.show()
