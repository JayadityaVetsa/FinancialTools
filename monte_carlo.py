import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
import base64
from io import BytesIO

def fetch_stock_data(symbol, start, end):
    try:
        stock_data = yf.download(symbol, start=start, end=end)
        if stock_data.empty:
            raise ValueError("No data fetched.")
        return stock_data
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
