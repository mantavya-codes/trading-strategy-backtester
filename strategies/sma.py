import numpy as np

def sma_strategy(df, short_window=20, long_window=50):
    df['SMA_short'] = df['Close'].rolling(window=short_window).mean()
    df['SMA_long'] = df['Close'].rolling(window=long_window).mean()

    df['Signal'] = np.where(df['SMA_short'] > df['SMA_long'], 1, 0)
    df['Position'] = df['Signal'].diff()

    return df