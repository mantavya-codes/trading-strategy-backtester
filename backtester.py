import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


import yfinance as yf

def load_data(symbol="BTC-USD", start="2021-01-01", end="2026-01-01"):
    df = yf.download(symbol, start=start, end=end)
    df.dropna(inplace=True)
    return df


def sma_strategy(df, short_window=20, long_window=50):
    df['SMA_short'] = df['Close'].rolling(window=short_window).mean()
    df['SMA_long'] = df['Close'].rolling(window=long_window).mean()

    df['Signal'] = np.where(df['SMA_short'] > df['SMA_long'], 1, 0)

    df['Position'] = df['Signal'].diff()

    return df



def backtest(df):
    df['Returns'] = df['Close'].pct_change()
    df['Strategy_Returns'] = df['Returns'] * df['Signal'].shift(1)

    cumulative_returns = (1 + df['Strategy_Returns']).cumprod()
    buy_hold = (1 + df['Returns']).cumprod()

    total_return = cumulative_returns.iloc[-1] - 1
    sharpe_ratio = np.sqrt(252) * (
        df['Strategy_Returns'].mean() / df['Strategy_Returns'].std()
    )

    drawdown = cumulative_returns / cumulative_returns.cummax() - 1
    max_drawdown = drawdown.min()

    print("Total Return: {:.2%}".format(total_return))
    print("Sharpe Ratio: {:.2f}".format(sharpe_ratio))
    print("Max Drawdown: {:.2%}".format(max_drawdown))

    return cumulative_returns, buy_hold




if __name__ == "__main__":
    data = load_data()
    data = sma_strategy(data)
    results, buy_hold = backtest(data)

    print("Final Strategy Return:", results.iloc[-1])

    plt.figure(figsize=(10, 5))
    plt.plot(results, label="Strategy")
    plt.plot(buy_hold, label="Buy & Hold")
    plt.legend()

    plt.title("Strategy vs Buy & Hold")
    plt.show()
