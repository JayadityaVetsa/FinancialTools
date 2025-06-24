import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import base64
from io import BytesIO
import requests
from datetime import datetime

def fetch_stock_data(symbol, start, end):
    try:
        # FMP API key - using the same one from main.py
        apikey_fmp = 'ZyymFyFsPSWfZBrL0skr97yoYyM5Czdr'
        
        # Format dates for API
        start_str = pd.to_datetime(start).strftime('%Y-%m-%d')
        end_str = pd.to_datetime(end).strftime('%Y-%m-%d')
        
        # Fetch data from FMP API
        url = f'https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}?apikey={apikey_fmp}&from={start_str}&to={end_str}'
        response = requests.get(url)
        data = response.json()
        
        if 'historical' not in data or not data['historical']:
            raise ValueError("No data fetched.")
            
        # Convert to DataFrame and format
        df = pd.DataFrame(data['historical'])
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        
        # Rename columns to match original format
        df = df.rename(columns={
            'close': 'Close',
            'high': 'High',
            'low': 'Low',
            'open': 'Open',
            'volume': 'Volume',
            'adjClose': 'Adj Close'
        })
        
        # Sort by date
        df = df.sort_index()
        
        return df
        
    except Exception as e:
        print(f"Error fetching stock data: {e}")
        return pd.DataFrame()

def monte_carlo_simulation(stock_data, num_simulations, num_days):
    try:
        log_returns = np.log(1 + stock_data['Adj Close'].pct_change())
        mean = log_returns.mean()
        variance = log_returns.var()
        drift = mean - (0.5 * variance)
        stdev = log_returns.std()
        daily_returns = np.exp(drift + stdev * np.random.randn(num_days, num_simulations))

        last_price = stock_data['Adj Close'].iloc[-1]
        price_paths = np.zeros_like(daily_returns)
        price_paths[0] = last_price

        for t in range(1, num_days):
            price_paths[t] = price_paths[t - 1] * daily_returns[t]

        return price_paths
    except Exception as e:
        print(f"Error during simulation: {e}")
        return None

def plot_simulation(symbol, price_paths, num_simulations, num_days):
    try:
        plt.figure(figsize=(10, 6))
        for i in range(num_simulations):
            plt.plot(price_paths[:, i], linewidth=0.7)

        plt.title(f'Monte Carlo Simulation for {symbol} Stock Price\n{num_simulations} simulations over {num_days} trading days')
        plt.xlabel('Days')
        plt.ylabel('Price')


        days = np.arange(num_days)
        avg_path = np.mean(price_paths, axis=1)
        poly_coeffs = np.polyfit(days, avg_path, 2)
        best_fit_line = np.polyval(poly_coeffs, days)
        plt.xlim([0, num_days - 1])  # Set x-axis limits explicitly
        plt.plot(days, best_fit_line, color='black', linestyle='--', linewidth=2, label='Line of Best Fit')

        final_prices = price_paths[-1]
        highest_point = np.max(final_prices)
        lowest_point = np.min(final_prices)

        plt.plot([0, num_days-1], [price_paths[0, 0], highest_point], color='green', linestyle='-', linewidth=2, label='Line to Highest Point')
        plt.plot([0, num_days-1], [price_paths[0, 0], lowest_point], color='red', linestyle='-', linewidth=2, label='Line to Lowest Point')

        plt.legend()

        buf = BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        plot_data = base64.b64encode(buf.getvalue()).decode('utf8')
        plt.close()

        return plot_data
    except Exception as e:
        print(f"Error generating plot: {e}")
        return None