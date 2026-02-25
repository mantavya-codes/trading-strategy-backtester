from strategies.sma import sma_strategy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf


def load_data(symbol="BTC-USD", start="2021-01-01"):
    df = yf.download(symbol, start=start)

    # Flatten multi-index columns if needed
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[['Close']]
    return df


def backtest(df, stop_loss=0.03, fee=0.001, verbose=True):
    df = df.copy()

    df['Position'] = df['Signal'].shift(1).fillna(0)

    entry_price = None
    highest_price = None
    in_position = False
    trades = []

    for i in range(len(df)):
        price = float(df['Close'].iloc[i])

        if df['Position'].iloc[i] == 1 and not in_position:
            entry_price = price
            highest_price = price
            in_position = True

        if in_position:
            if price > highest_price:
                highest_price = price

            if price <= highest_price * (1 - stop_loss):
                exit_price = price
                raw_return = (exit_price - entry_price) / entry_price
                trade_return = raw_return - (2 * fee)
                trades.append(trade_return)

                df.loc[df.index[i], 'Position'] = 0
                in_position = False
                entry_price = None
                highest_price = None

    if in_position and entry_price is not None:
        final_price = float(df['Close'].iloc[-1])
        raw_return = (final_price - entry_price) / entry_price
        trade_return = raw_return - (2 * fee)
        trades.append(trade_return)

    df['Strategy_Returns'] = df['Position'] * df['Close'].pct_change()
    df['Trade_Change'] = df['Position'].diff().abs()
    df['Strategy_Returns'] -= df['Trade_Change'] * fee
    df['Cumulative_Strategy'] = (1 + df['Strategy_Returns']).cumprod()

    cumulative_returns = df['Cumulative_Strategy']

    total_days = len(df)
    years = total_days / 252

    total_return = cumulative_returns.iloc[-1] - 1
    cagr = cumulative_returns.iloc[-1] ** (1 / years) - 1

    sharpe_ratio = np.sqrt(252) * (
        df['Strategy_Returns'].mean() /
        df['Strategy_Returns'].std()
    )

    drawdown = cumulative_returns / cumulative_returns.cummax() - 1
    max_drawdown = drawdown.min()

    exposure = df['Position'].mean()

    if verbose:
        print("Total Return: {:.2%}".format(total_return))
        print("CAGR: {:.2%}".format(cagr))
        print("Sharpe Ratio: {:.2f}".format(sharpe_ratio))
        print("Max Drawdown: {:.2%}".format(max_drawdown))
        print("Exposure: {:.2%}".format(exposure))

        if len(trades) > 0:
            trade_count = len(trades)
            win_rate = sum(1 for t in trades if t > 0) / trade_count
            avg_trade_return = sum(trades) / trade_count

            print("Trade Count:", trade_count)
            print("Win Rate: {:.2%}".format(win_rate))
            print("Average Trade Return: {:.2%}".format(avg_trade_return))

    return {
        "cagr": cagr,
        "max_dd": max_drawdown,
        "sharpe": sharpe_ratio,
        "exposure": exposure,
        "total_return": total_return,
    }
if __name__ == "__main__":
    import seaborn as sns

    symbol = "BTC-USD"
    data = load_data(symbol=symbol, start="2022-01-01")

    fast_range = range(10, 41, 5)   # 10,15,20,...40
    slow_range = range(50, 151, 10) # 50,60,...150

    heatmap_data = []

    for fast in fast_range:
        row = []
        for slow in slow_range:
            if fast >= slow:
                row.append(None)
                continue

            temp = sma_strategy(data.copy(), fast=fast, slow=slow)
            metrics = backtest(temp, stop_loss=0.03, verbose=False)

            score = metrics["cagr"] / abs(metrics["max_dd"])
            row.append(score)

        heatmap_data.append(row)

    import numpy as np
    heatmap_array = np.array(heatmap_data)

    plt.figure(figsize=(12, 6))
    sns.heatmap(
        heatmap_array,
        xticklabels=list(slow_range),
        yticklabels=list(fast_range),
        cmap="viridis"
    )

    plt.title("Return/Drawdown Heatmap")
    plt.xlabel("Slow SMA")
    plt.ylabel("Fast SMA")
    plt.show()