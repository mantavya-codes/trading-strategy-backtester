from strategies.sma import sma_strategy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf


# =============================
# Data Loader
# =============================
def load_data(symbol="BTC-USD", start="2021-01-01"):
    df = yf.download(symbol, start=start)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    return df[['Close']]


# =============================
# Backtest Engine
# =============================
def backtest(df, stop_loss=0.03, fee=0.001, verbose=True):

    df = df.copy()

    df['Position'] = df['Signal'].shift(1).fillna(0)

    entry_price = None
    highest_price = None
    in_position = False
    trades = []

    for i in range(len(df)):
        price = float(df['Close'].iloc[i])

        # Enter trade
        if df['Position'].iloc[i] == 1 and not in_position:
            entry_price = price
            highest_price = price
            in_position = True

        # Manage trade
        if in_position:
            if price > highest_price:
                highest_price = price

            # Trailing stop
            if price <= highest_price * (1 - stop_loss):
                raw_return = (price - entry_price) / entry_price
                trades.append(raw_return - 2 * fee)

                df.loc[df.index[i], 'Position'] = 0
                in_position = False
                entry_price = None
                highest_price = None

    # Close last trade
    if in_position and entry_price is not None:
        final_price = float(df['Close'].iloc[-1])
        raw_return = (final_price - entry_price) / entry_price
        trades.append(raw_return - 2 * fee)

    # Strategy returns
    df['Strategy_Returns'] = df['Position'] * df['Close'].pct_change()
    df['Trade_Change'] = df['Position'].diff().abs()
    df['Strategy_Returns'] -= df['Trade_Change'] * fee
    df['Cumulative_Strategy'] = (1 + df['Strategy_Returns']).cumprod()

    cumulative = df['Cumulative_Strategy']

    total_return = cumulative.iloc[-1] - 1
    years = len(df) / 252
    cagr = cumulative.iloc[-1] ** (1 / years) - 1

    # Safe Sharpe
    std = df['Strategy_Returns'].std()
    if std == 0 or np.isnan(std):
        sharpe_ratio = 0
    else:
        sharpe_ratio = np.sqrt(252) * (df['Strategy_Returns'].mean() / std)

    # Drawdown
    drawdown = cumulative / cumulative.cummax() - 1
    max_drawdown = drawdown.min()

    exposure = df['Position'].mean()

    if verbose:
        print(f"Total Return: {total_return:.2%}")
        print(f"CAGR: {cagr:.2%}")
        print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        print(f"Max Drawdown: {max_drawdown:.2%}")
        print(f"Exposure: {exposure:.2%}")

        if trades:
            win_rate = sum(t > 0 for t in trades) / len(trades)
            avg_trade = sum(trades) / len(trades)

            print("Trade Count:", len(trades))
            print(f"Win Rate: {win_rate:.2%}")
            print(f"Average Trade Return: {avg_trade:.2%}")

    return {
        "cagr": cagr,
        "max_dd": max_drawdown,
        "sharpe": sharpe_ratio,
        "exposure": exposure,
        "total_return": total_return,
    }


# =============================
# Parameter Optimization
# =============================
def optimize_sma(data, fast_range, slow_range):

    heatmap_data = []

    for fast in fast_range:
        row = []

        for slow in slow_range:

            if fast >= slow:
                row.append(np.nan)
                continue

            temp = sma_strategy(data.copy(), fast=fast, slow=slow)
            metrics = backtest(temp, stop_loss=0.03, verbose=False)

            # Safe score
            if metrics["max_dd"] == 0 or np.isnan(metrics["max_dd"]):
                score = 0
            else:
                score = metrics["cagr"] / abs(metrics["max_dd"])

            row.append(score)

        heatmap_data.append(row)

    heatmap_array = np.array(heatmap_data)

    best_idx = np.unravel_index(
        np.nanargmax(heatmap_array),
        heatmap_array.shape
    )

    best_fast = list(fast_range)[best_idx[0]]
    best_slow = list(slow_range)[best_idx[1]]

    return best_fast, best_slow


# =============================
# Walk Forward Validation
# =============================
if __name__ == "__main__":

    symbol = "BTC-USD"
    data = load_data(symbol=symbol, start="2022-01-01")

    walk_years = ["2023", "2024", "2025"]

    fast_range = range(10, 41, 5)
    slow_range = range(50, 151, 10)

    print("\n===== WALK FORWARD VALIDATION =====\n")

    for year in walk_years:

        print(f"\n--- Testing Year: {year} ---")

        train_data = data.loc["2022-01-01":f"{int(year)-1}-12-31"]
        test_data = data.loc[f"{year}-01-01":f"{year}-12-31"]

        if len(test_data) < 50:
            continue

        best_fast, best_slow = optimize_sma(
            train_data,
            fast_range,
            slow_range
        )

        print(f"Best Params -> Fast: {best_fast}, Slow: {best_slow}")

        test_temp = sma_strategy(
            test_data.copy(),
            fast=best_fast,
            slow=best_slow
        )

        print("Out-of-sample performance:")
        backtest(test_temp, stop_loss=0.03, verbose=True)