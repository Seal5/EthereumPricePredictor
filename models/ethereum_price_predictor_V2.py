import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
import xgboost as xgb

"""Load Data"""
data_path = './cleaned_dataset_2016_2020.csv'
df = pd.read_csv(data_path)
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Remove commas from numeric values and conver to float
for column in df.columns:
    if df[column].dtype == 'object':
        df[column] = df[column].str.replace(',', '').astype(float)
        
print(f"Data loaded successfully. Shape: {df.shape}")

"""Feature Engineering"""
# Select base features and create a copy
selected_features = ['USD_Price', 'BTC_High', 'ETH_High', 'Gold_USD', 'SPX_High']
df_with_features = df[selected_features].copy()

# Add 7-day moving averages (feature engineering)
df_with_features['SPX_7D_MA'] = df_with_features['SPX_High'].rolling(window=7).mean()
df_with_features['ETH_7D_MA'] = df_with_features['ETH_High'].rolling(window=7).mean()

# Remove rows first 6 days from rolling window
df_with_features.dropna(inplace=True)

"""Splitting and Scaling Data"""
# Initialize MinMaxScaler
scaler = MinMaxScaler()

# Scale all data together
scaled_data = scaler.fit_transform(df_with_features)
scaled_df = pd.DataFrame(scaled_data, columns=df_with_features.columns, index=df_with_features.index)

# Prepare features and target
target_column = 'ETH_High'
X = scaled_df.drop(columns=[target_column])
y = scaled_df[target_column]

# Split into train and test sets 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

"""Model Tuning and Training"""
# Perform hyperparameter tuning and train XGBoost model

# Parameter grid for model's Hyperparameters
param_grid = {
    'max_depth': [3, 6, 9],
    'learning_rate': [0.1, 0.05, 0.01],
    'n_estimators': [100, 200],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

# Create XGBoost regressor
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

# Create grid search
grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    scoring='neg_mean_squared_error',
    cv=3,
    verbose=1
)

# Fit the grid search
grid_search.fit(X_train, y_train)

# Get the best model
best_model = grid_search.best_estimator_

# Get the worst model
cv_results = pd.DataFrame(grid_search.cv_results_)
best_index = cv_results['rank_test_score'].argmin()
worst_index = cv_results['rank_test_score'].argmax()
worst_params = cv_results.loc[worst_index, 'params']

# Create and train the worst model with the identified parameters
worst_model = xgb.XGBRegressor(**worst_params, objective='reg:squarederror', random_state=42)
worst_model.fit(X_train, y_train)

# Print best and worst parameters
print("Best Model Parameters:")
print(grid_search.best_params_)
print("Worst Model Parameters:")
print(worst_params)

# Make predictions on the test set with both models
best_predictions = best_model.predict(X_test)
worst_predictions = worst_model.predict(X_test)

# Calculate and print error metrics for both models
best_mae = mean_absolute_error(y_test, best_predictions)
best_rmse = np.sqrt(mean_squared_error(y_test, best_predictions))
worst_mae = mean_absolute_error(y_test, worst_predictions)
worst_rmse = np.sqrt(mean_squared_error(y_test, worst_predictions))

print(f"Best Model - MAE: {best_mae:.4f}, RMSE: {best_rmse:.4f}")
print(f"Worst Model - MAE: {worst_mae:.4f}, RMSE: {worst_rmse:.4f}")

"""Visualization"""
def visualize_comparison(df, best_predictions, worst_predictions, y_test, scaler, test_indices):
    # Get the number of features and the index of the target column
    n_features = len(df.columns)
    eth_high_idx = df.columns.get_loc('ETH_High')
    
    # Inverse transform a single column's array
    def inv_transform(arr):
        dummy = np.zeros((len(arr), n_features))
        dummy[:, eth_high_idx] = arr
        return scaler.inverse_transform(dummy)[:, eth_high_idx]
    
    # Compute original scale values for actual and predicted
    actual = inv_transform(y_test.values)
    best_predicted = inv_transform(np.array(best_predictions))
    worst_predicted = inv_transform(np.array(worst_predictions))
    best_error = np.abs(actual - best_predicted)
    worst_error = np.abs(actual - worst_predicted)
    
    # Build a results DataFrame
    results_df = pd.DataFrame({
        'actual': actual,
        'best_predicted': best_predicted,
        'worst_predicted': worst_predicted,
        'best_error': best_error,
        'worst_error': worst_error
    }, index=test_indices)
    
    # Plot the actual vs predicted prices
    plt.figure(figsize=(16, 8))
    plt.plot(results_df.index, results_df['actual'], label='Actual ETH Prices', color='green')
    plt.plot(results_df.index, results_df['best_predicted'], label='Best Model Predictions', color='blue', linestyle='--')
    plt.plot(results_df.index, results_df['worst_predicted'], label='Worst Model Predictions', color='red', linestyle=':')
    plt.title('Ethereum Price Prediction - Best vs Worst Model Comparison')
    plt.xlabel('Date')
    plt.ylabel('Ethereum Price (High)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # Plot the prediction errors over time
    plt.figure(figsize=(16, 6))
    plt.plot(results_df.index, results_df['best_error'], label='Best Model Error', color='blue')
    plt.plot(results_df.index, results_df['worst_error'], label='Worst Model Error', color='red')
    plt.title('Prediction Error Comparison Over Time')
    plt.xlabel('Date')
    plt.ylabel('Absolute Error')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # Plot error distribution comparison
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.hist(results_df['best_error'], bins=20, alpha=0.7, color='blue')
    plt.title('Best Model Error Distribution')
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    
    plt.subplot(1, 2, 2)
    plt.hist(results_df['worst_error'], bins=20, alpha=0.7, color='red')
    plt.title('Worst Model Error Distribution')
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print("Best Model Prediction Statistics:")
    print(f"Average Error: ${results_df['best_error'].mean():.2f}")
    print(f"Max Error: ${results_df['best_error'].max():.2f}")
    print(f"Min Error: ${results_df['best_error'].min():.2f}")
    
    print("Worst Model Prediction Statistics:")
    print(f"Average Error: ${results_df['worst_error'].mean():.2f}")
    print(f"Max Error: ${results_df['worst_error'].max():.2f}")
    print(f"Min Error: ${results_df['worst_error'].min():.2f}")
    
    # Calculate percentage improvement
    improvement = ((results_df['worst_error'].mean() - results_df['best_error'].mean()) / 
                  results_df['worst_error'].mean()) * 100
    print(f"Average Error Improvement: {improvement:.2f}%")

# Call the comparison feature
visualize_comparison(df_with_features, best_predictions, worst_predictions, y_test, scaler, X_test.index)

# Observe the best model 
def visualize_results(df, predictions, y_test, scaler, test_indices):
    # Get the number of features and the index of the target column
    n_features = len(df.columns)
    eth_high_idx = df.columns.get_loc('ETH_High')
    
    # Inverse transform a single column's array
    def inv_transform(arr):
        dummy = np.zeros((len(arr), n_features))
        dummy[:, eth_high_idx] = arr
        return scaler.inverse_transform(dummy)[:, eth_high_idx]
    
    # Compute original scale values for actual and predicted
    actual = inv_transform(y_test.values)
    predicted = inv_transform(np.array(predictions))
    error = np.abs(actual - predicted)
    
    # Build a results DataFrame
    results_df = pd.DataFrame({
        'actual': actual,
        'predicted': predicted,
        'error': error
    }, index=test_indices)
    
    # Error Distribution
    plt.figure(figsize=(12, 6))
    plt.hist(error, bins=30, color='blue', alpha=0.7)
    plt.axvline(error.mean(), color='red', linestyle='--', label=f'Mean Error: ${error.mean():.2f}')
    plt.title('Error Distribution')
    plt.xlabel('Prediction Error ($)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Percentage Error Over Time
    plt.figure(figsize=(12, 6))
    pct_error = (error / results_df['actual']) * 100
    plt.plot(results_df.index, pct_error, color='red', alpha=0.7)
    plt.title('Percentage Error Over Time')
    plt.xlabel('Date')
    plt.ylabel('Percentage Error (%)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # Print detailed statistics
    print("Detailed Performance Metrics:")
    print(f"Root Mean Squared Error (RMSE): ${best_rmse:.2f}")
    print(f"Mean Absolute Error (MAE): ${best_mae:.2f}")
    print(f"Average Error: ${error.mean():.2f}")
    print(f"Median Error: ${np.median(error):.2f}")
    print(f"95th Percentile Error: ${np.percentile(error, 95):.2f}")

# Call the visualization function 
visualize_results(df_with_features, best_predictions, y_test, scaler, X_test.index)
