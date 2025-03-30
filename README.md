# Ethereum Price Predictor
A ML model that predicts Ehtereum prices using historical data and different market informations. The model uses XGBoost regression with optimized hyperparameters to provide price predictions for the next day. 

## Data Processing Pipeline

### Data Processing (`data_processing.py`)
- Combines multiple data sources into a single dataset
- Handles date parsing and formatting
- Outputs: `combined_dataset_2016_2020.csv`

### 2. Data Cleaning (`data_cleaning.py`)
- Fills missing weekend and holiday data
- Forward-fills market data using previous business day values
- Outputs: `cleaned_dataset_2016_2020.csv`

## Model Implementation (`ethereum_price_predictor_V2.py`)

## Features
Allows prediction of future Ethereum prices using:
- Ethereum price 
- Bitcoin price
- S&P 500 index
- Gold price
- USD worth
- Added lagged features and used feature engineered 7-day moving averages of [Making it a time series]:
  - S&P 500 index
  - Ethereum price

### Model Architecture
- Algorithm: XGBoost Regressor
- Hyperparameter optimization using GridSearchCV
- Train/Test split: 80/20 
- Data scaling using MinMaxScaler

### Visualization 
- Actual vs Predicted price comparisons
- Error distribution analysis
- Prediction error over time

## Performance Metrics
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- Percentage Error Analysis
- 95th Percentile Error

## Requirements

```python
pandas
numpy
matplotlib
scikit-learn
xgboost
```

## Usage

1. Process raw data:
```bash
python data_processing/data_processing.py
```

2. Clean the processed data:
```bash
python data_processing/data_cleaning.py
```

3. Run the prediction model:
```bash
python models/ethereum_price_predictor_V2.py
```







