import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt

# Load the dataset
file_path = r""
data = pd.read_csv(file_path)

# Prepare the data
X = data[['RH10', 'RH20', 'RH25', 'RH30', 'RH40', 'RH50', 'RH60','RH70', 'RH75', 'RH80', 'RH90', 'RH95', 'RH98']]  # predictors
y = data['Live_Biomass_Density']  # target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Visualize actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.title('Actual vs Predicted Values')
plt.xlabel('Actual Live Biomass Density (Mg/ha)')
plt.ylabel('Predicted Live Biomass Density (Mg/ha)')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)  # Diagonal line
plt.show()

# Calculate Mean Absolute Error
mae = mean_absolute_error(y_test, y_pred)

# Calculate R-squared
r2 = r2_score(y_test, y_pred)

# Calculate RMSE
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

# Calculate the average of actual values
average_actual = np.mean(y_test)

# Calculate Percentage RMSE
percentage_rmse = (rmse / average_actual) * 100


# Print the evaluation results
print(f'Mean Absolute Error (MAE): {mae:.2f} Mg/ha')
print(f'R-squared (R²): {r2:.2f}')
print(f'Percentage RMSE: {percentage_rmse:.2f}%')
print(f'Number of samples in the training set: {X_train.shape[0]}')
