import pandas as pd
import datetime as dt

# Read all CSV files with the necessary date parsing 
btc_df = pd.read_csv('./Data/BTC_USD.csv')

eth_df = pd.read_csv('./Data/ETH_day.csv')

gold_df = pd.read_csv('./Data/Gold.csv')

spx_df = pd.read_csv('./Data/SPX.csv')

usd_df = pd.read_csv('./Data/USDollar.csv')

# Process each dataframe to standardize dates and extract required columns

# For BTC
btc_df['Date'] = pd.to_datetime(btc_df.iloc[:, 0])  
btc_df = btc_df[['Date', 'High']].rename(columns={'High': 'BTC_High'})

# For ETH 
eth_df['Date'] = pd.to_datetime(eth_df.iloc[:, 0])  
eth_df = eth_df[['Date', 'High']].rename(columns={'High': 'ETH_High'})

# For Gold 
gold_df['Date'] = pd.to_datetime(gold_df.iloc[:, 0], format='%m/%d/%Y')
gold_df = gold_df[['Date', gold_df.columns[1]]].rename(columns={gold_df.columns[1]: 'Gold_USD'})

# For SPX (has YYYY-MM-DD format)
spx_df['Date'] = pd.to_datetime(spx_df.iloc[:, 0])
spx_df = spx_df[['Date', 'High']].rename(columns={'High': 'SPX_High'})

# For USD Dollar 
usd_df['Date'] = pd.to_datetime(usd_df.iloc[:, 0].str.replace('"', ''))
usd_df['USD_Price'] = usd_df.iloc[:, 1]  
usd_df = usd_df[['Date', 'USD_Price']]

# Filter each dataframe for the date range
start_date = pd.to_datetime('2016-05-09')
end_date = pd.to_datetime('2020-01-01')

btc_df = btc_df[(btc_df['Date'] >= start_date) & (btc_df['Date'] <= end_date)]
eth_df = eth_df[(eth_df['Date'] >= start_date) & (eth_df['Date'] <= end_date)]
gold_df = gold_df[(gold_df['Date'] >= start_date) & (gold_df['Date'] <= end_date)]
spx_df = spx_df[(spx_df['Date'] >= start_date) & (spx_df['Date'] <= end_date)]
usd_df = usd_df[(usd_df['Date'] >= start_date) & (usd_df['Date'] <= end_date)]

# Merge all dataframes on the Date column
combined_df = usd_df

# Merge each dataframe
combined_df = pd.merge(combined_df, btc_df, on='Date', how='outer')
combined_df = pd.merge(combined_df, eth_df, on='Date', how='outer')
combined_df = pd.merge(combined_df, gold_df, on='Date', how='outer')
combined_df = pd.merge(combined_df, spx_df, on='Date', how='outer')

# Sort by date
combined_df = combined_df.sort_values('Date')

# Check for missing values
missing_values = combined_df.isnull().sum()
print("Missing values in each column:")
print(missing_values)

# Save the combined dataset
combined_df.to_csv('combined_dataset_2016_2020.csv', index=False, date_format='%Y-%m-%d')

print("Combined dataset created successfully")
