import numpy as np

def sma_strategy(df, fast=20, slow=50):
    df = df.copy()

    df['SMA_Fast'] = df['Close'].rolling(fast).mean()
    df['SMA_Slow'] = df['Close'].rolling(slow).mean()

    df['Signal'] = np.where(df['SMA_Fast'] > df['SMA_Slow'], 1, 0)

    return df