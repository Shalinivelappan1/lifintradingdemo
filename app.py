import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Strategy Learning Lab", layout="wide")
st.title("📚 Strategy Learning Lab")
st.write("Upload any stock/index CSV with Date and Close columns")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:

    # ================= LOAD DATA =================
    df = pd.read_csv(uploaded_file)
    df.columns = [c.strip() for c in df.columns]

    date_col = [c for c in df.columns if "date" in c.lower()][0]
    price_col = [c for c in df.columns if "close" in c.lower() or "price" in c.lower()][-1]

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")

    df = df.dropna().sort_values(date_col).reset_index(drop=True)

    if len(df) < 50:
        st.warning("Dataset is short. Some strategies may not trigger.")

    # ================= SIDEBAR =================
    st.sidebar.header("Controls")

    strategy = st.sidebar.selectbox(
        "Select Strategy",
        [
            "Buy & Hold",
            "Momentum (Single MA)",
            "Dual MA Crossover",
            "Mean Reversion",
            "RSI Reversal",
            "Volatility Breakout",
            "Trend Pullback",
            "Blended Strategy"
        ]
    )

    ma_short = st.sidebar.slider("Short MA", 10, 100, 50)
    ma_long = st.sidebar.slider("Long MA", 100, 300, 200)
    deviation = st.sidebar.slider("Mean Reversion %", 1, 10, 3)
    rsi_period = st.sidebar.slider("RSI Period", 5, 30, 14)
    rsi_level = st.sidebar.slider("RSI Oversold", 10, 40, 30)

    # ================= INDICATORS =================
    df["MA_S"] = df[price_col].rolling(ma_short).mean()
    df["MA_L"] = df[price_col].rolling(ma_long).mean()

    delta = df[price_col].diff()
    gain = delta.where(delta > 0, 0).rolling(rsi_period).mean()
    loss = -delta.where(delta < 0, 0).rolling(rsi_period).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    df["Return"] = df[price_col].pct_change()

    # ================= STRATEGIES =================
    def buy_hold():
        return pd.Series(1, index=df.index)

    def momentum():
        return pd.Series(np.where(df[price_col] > df["MA_S"], 1, 0), index=df.index)

    def dual_ma():
        return pd.Series(np.where(df["MA_S"] > df["MA_L"], 1, 0), index=df.index)

    def mean_reversion():
        dev = (df[price_col] - df["MA_S"]) / df["MA_S"]
        return pd.Series(np.where(dev < -deviation/100, 1, 0), index=df.index)

    def rsi_strategy():
        return pd.Series(np.where(df["RSI"] < rsi_level, 1, 0), index=df.index)

    def breakout():
        br = df[price_col] > df[price_col].rolling(20).max().shift(1)
        return pd.Series(np.where(br, 1, 0), index=df.index)

    def trend_pullback():
        uptrend = df["MA_S"] > df["MA_L"]
        pullback = df[price_col] < df["MA_S"]
        return pd.Series(np.where(uptrend & pullback, 1, 0), index=df.index)

    def blended():
        return (momentum() + dual_ma() + mean_reversion() + rsi_strategy() + breakout()) / 5

    strategy_map = {
        "Buy & Hold": buy_hold(),
        "Momentum (Single MA)": momentum(),
        "Dual MA Crossover": dual_ma(),
        "Mean Reversion": mean_reversion(),
        "RSI Reversal": rsi_strategy(),
        "Volatility Breakout": breakout(),
        "Trend Pullback": trend_pullback(),
        "Blended Strategy": blended(),
    }

    # ================= APPLY SELECTED STRATEGY =================
    position = strategy_map[strategy]
    df["Position"] = position
    df["Strat_Return"] = df["Return"] * df["Position"].shift(1)
    df["Equity"] = (1 + df["Strat_Return"].fillna(0)).cumprod()

    # ================= EXPLANATION =================
    st.subheader("📖 Strategy Explanation")

    explanation = {
        "Buy & Hold": "Always invested. Shows long-term compounding.",
        "Momentum (Single MA)": "Buy when price above moving average.",
        "Dual MA Crossover": "Buy when short MA crosses long MA.",
        "Mean Reversion": "Buy deep dips expecting bounce.",
        "RSI Reversal": "Buy when RSI oversold.",
        "Volatility Breakout": "Buy new highs.",
        "Trend Pullback": "Buy pullbacks in uptrend.",
        "Blended Strategy": "Mix of multiple strategies across time."
    }

    st.info(explanation[strategy])

    # ================= CHART =================
    entries = df[(df["Position"] == 1) & (df["Position"].shift(1) == 0)]
    exits = df[(df["Position"] == 0) & (df["Position"].shift(1) == 1)]

    if strategy == "RSI Reversal":
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,7), sharex=True)
        ax1.plot(df[date_col], df[price_col])
        ax1.scatter(entries[date_col], entries[price_col], marker="^")
        ax1.scatter(exits[date_col], exits[price_col], marker="v")
        ax2.plot(df[date_col], df["RSI"])
        ax2.axhline(rsi_level, linestyle="--")
        ax2.axhline(70, linestyle="--")
        st.pyplot(fig)

    elif strategy == "Volatility Breakout":
        fig, ax = plt.subplots(figsize=(10,5))
        band = df[price_col].rolling(20).max()
        ax.plot(df[date_col], df[price_col])
        ax.plot(df[date_col], band, linestyle="--")
        ax.scatter(entries[date_col], entries[price_col], marker="^")
        ax.scatter(exits[date_col], exits[price_col], marker="v")
        st.pyplot(fig)

    else:
        fig, ax = plt.subplots(figsize=(10,5))
        ax.plot(df[date_col], df[price_col])
        ax.plot(df[date_col], df["MA_S"])
        if strategy in ["Dual MA Crossover", "Trend Pullback"]:
            ax.plot(df[date_col], df["MA_L"])
        ax.scatter(entries[date_col], entries[price_col], marker="^")
        ax.scatter(exits[date_col], exits[price_col], marker="v")
        st.pyplot(fig)

    # ================= WHAT TO OBSERVE =================
    st.subheader("🎓 What to Observe")

    learnings = {
        "Buy & Hold": "Observe compounding and drawdowns.",
        "Momentum (Single MA)": "Watch whipsaws in sideways markets.",
        "Dual MA Crossover": "Notice delayed entries.",
        "Mean Reversion": "Fails in strong downtrends.",
        "RSI Reversal": "Frequent signals.",
        "Volatility Breakout": "Few but strong trades.",
        "Trend Pullback": "Better timing in trends.",
        "Blended Strategy": "Smoother curve from diversification."
    }

    st.write(learnings[strategy])

    # ================= SAFE METRICS =================
    st.subheader("📊 Strategy Metrics")

    if df["Equity"].dropna().shape[0] > 0:

        total_return = (df["Equity"].iloc[-1] - 1) * 100
        max_dd = (df["Equity"] / df["Equity"].cummax() - 1).min() * 100
        sharpe = df["Strat_Return"].mean()/df["Strat_Return"].std()*np.sqrt(252) if df["Strat_Return"].std()!=0 else 0
        trades = (df["Position"].diff()==1).sum()

        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Return %", f"{total_return:.2f}")
        c2.metric("Sharpe", f"{sharpe:.2f}")
        c3.metric("MaxDD %", f"{max_dd:.2f}")
        c4.metric("Trades", int(trades))

    else:
        st.warning("No trades generated for this strategy & parameters.")

    # ================= COMPARISON TABLE =================
    st.subheader("📊 Multi-Strategy Comparison")

    results = []

    for name, pos in strategy_map.items():

        strat_ret = df["Return"] * pos.shift(1)
        equity = (1 + strat_ret.fillna(0)).cumprod()

        if equity.dropna().shape[0] == 0:
            results.append([name, 0, 0, 0, 0])
            continue

        ret = (equity.iloc[-1] - 1) * 100
        dd = (equity/equity.cummax() - 1).min() * 100
        sharpe_val = strat_ret.mean()/strat_ret.std()*np.sqrt(252) if strat_ret.std()!=0 else 0
        trades_val = (pos.diff()==1).sum()

        results.append([name, round(ret,2), round(sharpe_val,2), round(dd,2), int(trades_val)])

    comp = pd.DataFrame(results, columns=["Strategy","Return %","Sharpe","MaxDD %","Trades"])
    st.dataframe(comp)

    # ================= EQUITY COMPARISON =================
    st.subheader("📈 Equity Curve Comparison")

    fig2, ax2 = plt.subplots(figsize=(10,5))

    for name, pos in strategy_map.items():
        strat_ret = df["Return"] * pos.shift(1)
        equity = (1 + strat_ret.fillna(0)).cumprod()
        ax2.plot(df[date_col], equity, label=name)

    ax2.legend()
    st.pyplot(fig2)

else:
    st.info("Upload CSV to begin.")
