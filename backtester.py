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


def backtest(df, stop_loss=0.04, fee=0.001):
    df = df.copy()

    # Shift signal to create position
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

        # Manage open trade
        if in_position:

            # Update highest price
            if price > highest_price:
                highest_price = price

            # Trailing stop
            if price <= highest_price * (1 - stop_loss):
                exit_price = price
                raw_return = (exit_price - entry_price) / entry_price
                trade_return = raw_return - (2 * fee)
                trades.append(trade_return)

                df.loc[df.index[i], 'Position'] = 0
                in_position = False
                entry_price = None
                highest_price = None

    # Close last open trade at final price
    if in_position and entry_price is not None:
        final_price = float(df['Close'].iloc[-1])
        raw_return = (final_price - entry_price) / entry_price
        trade_return = raw_return - (2 * fee)
        trades.append(trade_return)

    # Strategy returns
    df['Strategy_Returns'] = df['Position'] * df['Close'].pct_change()
    df['Trade_Change'] = df['Position'].diff().abs()
    df['Strategy_Returns'] -= df['Trade_Change'] * fee
    df['Cumulative_Strategy'] = (1 + df['Strategy_Returns']).cumprod()
    

    cumulative_returns = df['Cumulative_Strategy']
   # Calculate number of years
    total_days = len(df)
    years = total_days / 252  # 252 trading days per year

# Performance metrics
    total_return = cumulative_returns.iloc[-1] - 1
    cagr = cumulative_returns.iloc[-1] ** (1 / years) - 1

    sharpe_ratio = np.sqrt(252) * (
    df['Strategy_Returns'].mean() /
    df['Strategy_Returns'].std()
)

    drawdown = cumulative_returns / cumulative_returns.cummax() - 1
    max_drawdown = drawdown.min()

    print("Total Return: {:.2%}".format(total_return))
    print("CAGR: {:.2%}".format(cagr))
    print("Sharpe Ratio: {:.2f}".format(sharpe_ratio))
    print("Max Drawdown: {:.2%}".format(max_drawdown))

    # Exposure %
    exposure = df['Position'].mean()
    print("Exposure: {:.2%}".format(exposure))

    # Trade analytics
    if len(trades) > 0:
        trade_count = len(trades)
        win_rate = sum(1 for t in trades if t > 0) / trade_count
        avg_trade_return = sum(trades) / trade_count

        print("Trade Count:", trade_count)
        print("Win Rate: {:.2%}".format(win_rate))
        print("Average Trade Return: {:.2%}".format(avg_trade_return))

    # Buy & Hold curve
    df['Buy_Hold'] = (1 + df['Close'].pct_change()).cumprod()

    return {
        "cagr": cagr,   
        "max_dd": max_drawdown,
        "sharpe": sharpe_ratio,
        "exposure": exposure,
        "total_return": total_return,
    }


if __name__ == "__main__":
    symbol = "BTC-USD"
    stop_values = [0.03, 0.04, 0.05, 0.06, 0.07]

    results_table = []

    for stop in stop_values:
        print(f"\nTesting stop_loss = {stop}")

        data = load_data(symbol=symbol, start="2022-01-01")
        data = sma_strategy(data)

        metrics = backtest(data, stop_loss=stop)

        score = metrics["cagr"] / abs(metrics["max_dd"])

        results_table.append({
            "stop": stop,
            "cagr": metrics["cagr"],
            "max_dd": metrics["max_dd"],
            "score": score
        })

    print("\n=== Optimization Results ===")

    for row in sorted(results_table, key=lambda x: x["score"], reverse=True):
        print(
            f"Stop: {row['stop']} | "
            f"CAGR: {row['cagr']:.2%} | "
            f"MaxDD: {row['max_dd']:.2%} | "
            f"Score: {row['score']:.2f}"
        )
        
    #plt.figure(figsize=(10, 5))
    #plt.plot(results, label="Strategy")
    #plt.plot(buy_hold, label="Buy & Hold")
    #plt.legend()
    #plt.title("Strategy vs Buy & Hold")
    #plt.show()