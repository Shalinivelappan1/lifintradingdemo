import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Trading Strategy Learning Lab", layout="wide")
st.title("📚 Trading Strategy Learning Lab")

st.write("Upload Nifty historical CSV and explore how strategies behave.")

uploaded_file = st.file_uploader("Upload Nifty CSV", type=["csv"])

if uploaded_file:

    df = pd.read_csv(uploaded_file)
    df.columns = [c.strip() for c in df.columns]

    date_col = [c for c in df.columns if "Date" in c or "date" in c][0]
    price_col = [c for c in df.columns if "Close" in c or "Price" in c][-1]

    df[date_col] = pd.to_datetime(df[date_col], dayfirst=True, errors="coerce")
    df[price_col] = df[price_col].astype(str).str.replace(",", "")
    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
    df = df.dropna().sort_values(date_col).reset_index(drop=True)

    # ========= SIDEBAR =========
    st.sidebar.header("Controls")

    strategy = st.sidebar.selectbox(
        "Choose Strategy",
        [
            "Buy & Hold",
            "Momentum (Single MA)",
            "Dual MA Crossover",
            "Mean Reversion",
            "RSI Reversal",
            "Volatility Breakout",
            "Trend Pullback"
        ]
    )

    ma_short = st.sidebar.slider("Short MA", 10, 100, 50)
    ma_long = st.sidebar.slider("Long MA", 100, 300, 200)
    deviation = st.sidebar.slider("Mean Reversion %", 1, 10, 3)
    rsi_period = st.sidebar.slider("RSI Period", 5, 30, 14)
    rsi_level = st.sidebar.slider("RSI Oversold", 10, 40, 30)

    # ========= INDICATORS =========
    df["MA_S"] = df[price_col].rolling(ma_short).mean()
    df["MA_L"] = df[price_col].rolling(ma_long).mean()

    delta = df[price_col].diff()
    gain = delta.where(delta > 0, 0).rolling(rsi_period).mean()
    loss = -delta.where(delta < 0, 0).rolling(rsi_period).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    df["Position"] = 0

    # ========= STRATEGY LOGIC =========

    if strategy == "Buy & Hold":
        st.subheader("📖 Buy & Hold")
        st.write("Always invested. Benchmark strategy.")
        df["Position"] = 1

    elif strategy == "Momentum (Single MA)":
        st.subheader("📖 Momentum Strategy")
        st.write("Buy when price above moving average.")
        df["Position"] = np.where(df[price_col] > df["MA_S"], 1, 0)

    elif strategy == "Dual MA Crossover":
        st.subheader("📖 Dual MA Strategy")
        st.write("Buy when short MA crosses above long MA.")
        df["Position"] = np.where(df["MA_S"] > df["MA_L"], 1, 0)

    elif strategy == "Mean Reversion":
        st.subheader("📖 Mean Reversion")
        st.write("Buy sharp dips expecting bounce.")
        dev = (df[price_col] - df["MA_S"]) / df["MA_S"]
        df["Position"] = np.where(dev < -deviation/100, 1, 0)

    elif strategy == "RSI Reversal":
        st.subheader("📖 RSI Strategy")
        st.write("Buy when RSI oversold.")
        df["Position"] = np.where(df["RSI"] < rsi_level, 1, 0)

    elif strategy == "Volatility Breakout":
        st.subheader("📖 Breakout Strategy")
        st.write("Buy when price breaks 20-day high.")
        breakout = df[price_col] > df[price_col].rolling(20).max().shift(1)
        df["Position"] = np.where(breakout, 1, 0)

    elif strategy == "Trend Pullback":
        st.subheader("📖 Trend + Pullback")
        st.write("Buy pullbacks in uptrend.")
        uptrend = df["MA_S"] > df["MA_L"]
        pullback = df[price_col] < df["MA_S"]
        df["Position"] = np.where(uptrend & pullback, 1, 0)

    # ========= RETURNS =========
    df["Return"] = df[price_col].pct_change()
    df["Strat_Return"] = df["Return"] * df["Position"].shift(1)
    df["Equity"] = (1 + df["Strat_Return"].fillna(0)).cumprod()

    # ========= ENTRY EXIT =========
    entries = df[(df["Position"]==1)&(df["Position"].shift(1)==0)]
    exits = df[(df["Position"]==0)&(df["Position"].shift(1)==1)]

    # ========= CHART =========
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(df[date_col], df[price_col], label="Price")
    ax.plot(df[date_col], df["MA_S"], label="MA Short")
    ax.plot(df[date_col], df["MA_L"], label="MA Long", alpha=0.5)

    ax.scatter(entries[date_col], entries[price_col], marker="^", label="Entry")
    ax.scatter(exits[date_col], exits[price_col], marker="v", label="Exit")

    ax.legend()
    st.pyplot(fig)

    # ========= METRICS =========
    total_return = (df["Equity"].iloc[-1] - 1) * 100
    st.metric("Strategy Return %", f"{total_return:.2f}")

    # ========= MULTI STRATEGY COMPARISON =========
    st.subheader("📊 Multi-Strategy Comparison")

    def run_strategy(pos):
        r = df["Return"] * pos.shift(1)
        equity = (1+r.fillna(0)).cumprod()
        ret = (equity.iloc[-1]-1)*100
        dd = (equity/equity.cummax()-1).min()*100
        sharpe = r.mean()/r.std()*np.sqrt(252) if r.std()!=0 else 0
        trades = (pos.diff()==1).sum()
        return ret, sharpe, dd, trades

    results = []

    strategies = {
        "BuyHold": pd.Series(1, index=df.index),
        "Momentum": pd.Series(np.where(df[price_col]>df["MA_S"],1,0)),
        "DualMA": pd.Series(np.where(df["MA_S"]>df["MA_L"],1,0)),
        "RSI": pd.Series(np.where(df["RSI"]<rsi_level,1,0))
    }

    for name,pos in strategies.items():
        r,s,d,t = run_strategy(pos)
        results.append([name, round(r,2), round(s,2), round(d,2), int(t)])

    comp = pd.DataFrame(results, columns=["Strategy","Return %","Sharpe","MaxDD %","Trades"])
    st.dataframe(comp)

    # ========= EQUITY COMPARISON =========
    st.subheader("📈 Equity Curve Comparison")
    fig2, ax2 = plt.subplots()

    for name,pos in strategies.items():
        r = df["Return"]*pos.shift(1)
        equity = (1+r.fillna(0)).cumprod()
        ax2.plot(df[date_col], equity, label=name)

    ax2.legend()
    st.pyplot(fig2)

else:
    st.info("Upload CSV to begin.")
