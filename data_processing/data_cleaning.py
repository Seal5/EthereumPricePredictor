import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Clean the dataset through filling in the weekend data and holiday data using the previous business day values
def clean_dataset(file_path="./combined_dataset_2016_2020.csv"):
    # Load the dataset
    print(f"Loading dataset")
    df = pd.read_csv(file_path)
    
    # Must convert all the date into datetime format 
    df['Date'] = pd.to_datetime(df['Date'])
    
    print("Filling in weekend and holiday data")
    
    # Create a continuous date range to detect missing dates
    date_range = pd.date_range(start=df['Date'].min(), end=df['Date'].max())
    date_range_df = pd.DataFrame({'Date': date_range})
    
    # Merge with the original dataframe to include any missing dates
    df = pd.merge(date_range_df, df, on='Date', how='left')
    
    # Forward fill for market data columns handling weekends and holidays
    market_columns = ['USD_Price', 'SPX_High', 'Gold_USD']
    
    # First, forward fill to handle weekends and holidays using previous business days' data
    for col in market_columns:
        df[col] = df[col].ffill()
    
    # Save the cleaned dataset
    output_file = "cleaned_dataset_2016_2020.csv"
    print(f"Saving cleaned dataset to as {output_file}")
    df.to_csv(output_file, index=False)
    
    print("Data cleaning completed")
    return df


if __name__ == "__main__":
    # Run the data cleaning 
    cleaned_df = clean_dataset()
    