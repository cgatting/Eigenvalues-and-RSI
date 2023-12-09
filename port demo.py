import yfinance as yf
import pandas as pd
import numpy as np

def download_stock_data(tickers, start_date, end_date):
    """Download historical adjusted close prices for given tickers."""
    return yf.download(tickers, start=start_date, end=end_date)['Adj Close']

def calculate_portfolio_metrics(data):
    """Calculate returns, covariance matrix, eigenvalues, and eigenvectors."""
    returns = data.pct_change().dropna()
    cov_matrix = returns.cov()
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    eigenvectors_normalized = eigenvectors / np.linalg.norm(eigenvectors, axis=0)
    return returns, eigenvalues, eigenvectors_normalized

def calculate_rsi(data, window=14):
    """Calculate the Relative Strength Index (RSI) for the given data."""
    delta = data.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def simulate_trading(data, tickers, initial_investment, rsi_threshold_buy, rsi_threshold_sell, eigenvalues, eigenvectors_normalized):
    """Simulate trading based on RSI and eigendecomposition."""
    portfolio = {ticker: 0 for ticker in tickers}
    cash_balance = initial_investment
    for date in data.index:
        rsi_values = calculate_rsi(data.loc[:date].iloc[-15:])
        eigenvalue_proportions = eigenvalues / eigenvalues.sum()
        lot_sizes = (initial_investment * eigenvalue_proportions) / data.loc[date]
        for i, ticker in enumerate(tickers):
            if rsi_values[ticker].iloc[-1] < rsi_threshold_buy and cash_balance > 0:
                print("BUY")
                num_shares_to_buy = int(cash_balance / lot_sizes[i])
                cost = num_shares_to_buy * data[ticker].loc[date]
                cash_balance -= cost
                portfolio[ticker] += num_shares_to_buy
            elif rsi_values[ticker].iloc[-1] > rsi_threshold_sell and portfolio[ticker] > 0:
                print("SELL")
                num_shares_to_sell = min(portfolio[ticker], int(lot_sizes[i]))
                cash_balance += num_shares_to_sell * data[ticker].loc[date]
                portfolio[ticker] -= num_shares_to_sell
    portfolio_value = cash_balance + sum(portfolio[ticker] * data[ticker].iloc[-1] for ticker in tickers)
    return portfolio, portfolio_value

def main():
    # Parameters
    tickers = ["AAPL", "MSFT", "GOOGL"]
    start_date = "2020-01-01"
    end_date = "2023-01-01"
    initial_investment = 1000
    rsi_threshold_buy = 20
    rsi_threshold_sell = 70
    # Fetch and process data
    data = download_stock_data(tickers, start_date, end_date)
    returns, eigenvalues, eigenvectors_normalized = calculate_portfolio_metrics(data)
    print(eigenvectors_normalized)
    # Simulate trading
    portfolio, portfolio_value = simulate_trading(data, tickers, initial_investment, rsi_threshold_buy, rsi_threshold_sell, eigenvalues, eigenvectors_normalized)

    # Display results
    print(f"Final Portfolio Value: {round(portfolio_value, 2)}")
    print("Portfolio Holdings:")
    for ticker, shares in portfolio.items():
        print(f"{ticker}: {shares} shares")

if __name__ == "__main__":
    main()
