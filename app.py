import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Trading Strategy Learning Lab", layout="wide")
st.title("📚 Trading Strategy Learning Lab")
st.write("Upload Nifty historical CSV and explore how strategies behave.")

uploaded_file = st.file_uploader("Upload Nifty CSV", type=["csv"])

if uploaded_file:

    # ================= DATA LOAD =================
    df = pd.read_csv(uploaded_file)
    df.columns = [c.strip() for c in df.columns]

    date_col = [c for c in df.columns if "Date" in c or "date" in c][0]
    price_col = [c for c in df.columns if "Close" in c or "Price" in c][-1]

    df[date_col] = pd.to_datetime(df[date_col], dayfirst=True, errors="coerce")
    df[price_col] = df[price_col].astype(str).str.replace(",", "")
    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")

    df = df.dropna().sort_values(date_col).reset_index(drop=True)

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
            "Trend Pullback"
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

    # ================= STRATEGY FUNCTIONS =================
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

    strategy_map = {
        "Buy & Hold": buy_hold(),
        "Momentum (Single MA)": momentum(),
        "Dual MA Crossover": dual_ma(),
        "Mean Reversion": mean_reversion(),
        "RSI Reversal": rsi_strategy(),
        "Volatility Breakout": breakout(),
        "Trend Pullback": trend_pullback(),
    }

    # ================= SELECT STRATEGY =================
    position = strategy_map[strategy]
    df["Position"] = position
    df["Strat_Return"] = df["Return"] * df["Position"].shift(1)
    df["Equity"] = (1 + df["Strat_Return"].fillna(0)).cumprod()

    # ================= EXPLANATION PANEL =================
    st.subheader("📖 Strategy Explanation")

    explanation_text = {
        "Buy & Hold": "Always invested. Demonstrates long-term compounding and drawdowns.",
        "Momentum (Single MA)": "Buy when price above MA. Fast signals, more noise.",
        "Dual MA Crossover": "Buy when short MA crosses long MA. Slower but smoother.",
        "Mean Reversion": "Buy deep dips below average. Works in range, fails in crashes.",
        "RSI Reversal": "Buy when RSI oversold. Many short-term signals.",
        "Volatility Breakout": "Buy new highs. Captures strong momentum moves.",
        "Trend Pullback": "Buy pullbacks in established uptrend."
    }

    st.info(explanation_text[strategy])

    # ================= STRATEGY-SPECIFIC CHART =================
    entries = df[(df["Position"] == 1) & (df["Position"].shift(1) == 0)]
    exits = df[(df["Position"] == 0) & (df["Position"].shift(1) == 1)]

    if strategy == "RSI Reversal":
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,7), sharex=True)

        ax1.plot(df[date_col], df[price_col], label="Price")
        ax1.scatter(entries[date_col], entries[price_col], marker="^")
        ax1.scatter(exits[date_col], exits[price_col], marker="v")
        ax1.legend()

        ax2.plot(df[date_col], df["RSI"], label="RSI")
        ax2.axhline(rsi_level, linestyle="--")
        ax2.axhline(70, linestyle="--")
        ax2.legend()

        st.pyplot(fig)

    elif strategy == "Volatility Breakout":
        fig, ax = plt.subplots(figsize=(10,5))
        rolling_high = df[price_col].rolling(20).max()
        ax.plot(df[date_col], df[price_col], label="Price")
        ax.plot(df[date_col], rolling_high, linestyle="--", label="20-Day High")
        ax.scatter(entries[date_col], entries[price_col], marker="^")
        ax.scatter(exits[date_col], exits[price_col], marker="v")
        ax.legend()
        st.pyplot(fig)

    else:
        fig, ax = plt.subplots(figsize=(10,5))
        ax.plot(df[date_col], df[price_col], label="Price")
        ax.plot(df[date_col], df["MA_S"], label="Short MA")

        if strategy in ["Dual MA Crossover", "Trend Pullback"]:
            ax.plot(df[date_col], df["MA_L"], label="Long MA")

        ax.scatter(entries[date_col], entries[price_col], marker="^")
        ax.scatter(exits[date_col], exits[price_col], marker="v")
        ax.legend()
        st.pyplot(fig)

    # ================= WHAT TO OBSERVE =================
    st.subheader("🎓 What Students Should Observe")

    learnings = {
        "Buy & Hold": "Notice compounding and large drawdowns.",
        "Momentum (Single MA)": "Frequent trades. Early but noisy entries.",
        "Dual MA Crossover": "Fewer trades. Late entry but smoother trend.",
        "Mean Reversion": "Works in sideways markets. Fails in strong trends.",
        "RSI Reversal": "Many false signals in strong downtrends.",
        "Volatility Breakout": "Few trades but strong sustained moves.",
        "Trend Pullback": "Combines trend direction with better timing."
    }

    st.write(learnings[strategy])

    # ================= METRICS =================
    total_return = (df["Equity"].iloc[-1] - 1) * 100
    max_dd = (df["Equity"] / df["Equity"].cummax() - 1).min() * 100
    sharpe = df["Strat_Return"].mean()/df["Strat_Return"].std()*np.sqrt(252) if df["Strat_Return"].std()!=0 else 0
    trades = (df["Position"].diff()==1).sum()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Return %", f"{total_return:.2f}")
    col2.metric("Sharpe", f"{sharpe:.2f}")
    col3.metric("MaxDD %", f"{max_dd:.2f}")
    col4.metric("Trades", int(trades))

    # ================= COMPARISON TABLE =================
    st.subheader("📊 Multi-Strategy Comparison")

    results = []

    for name, pos in strategy_map.items():
        strat_ret = df["Return"] * pos.shift(1)
        equity = (1 + strat_ret.fillna(0)).cumprod()
        ret = (equity.iloc[-1] - 1) * 100
        dd = (equity/equity.cummax() - 1).min() * 100
        sharpe_val = strat_ret.mean()/strat_ret.std()*np.sqrt(252) if strat_ret.std()!=0 else 0
        trades_val = (pos.diff()==1).sum()

        results.append([name, round(ret,2), round(sharpe_val,2), round(dd,2), int(trades_val)])

    comp = pd.DataFrame(results, columns=["Strategy","Return %","Sharpe","MaxDD %","Trades"])
    st.dataframe(comp)

    # ================= EQUITY CURVE =================
    st.subheader("📈 Equity Curve Comparison")

    fig2, ax2 = plt.subplots(figsize=(10,5))
    for name, pos in strategy_map.items():
        strat_ret = df["Return"] * pos.shift(1)
        equity = (1 + strat_ret.fillna(0)).cumprod()
        ax2.plot(df[date_col], equity, label=name)

    ax2.legend()
    st.pyplot(fig2)

else:
    st.info("Upload Nifty CSV to begin.")
