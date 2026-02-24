import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Trading Strategy Lab", layout="wide")

st.title("📈 Nifty Trading Strategy Classroom Lab")
st.write("Upload Nifty historical CSV and test multiple trading strategies.")

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

    # Sidebar Controls
    st.sidebar.header("Strategy Settings")

    strategy = st.sidebar.selectbox(
        "Select Strategy",
        [
            "Buy & Hold",
            "Momentum (Price > MA)",
            "Dual MA Crossover",
            "Mean Reversion",
            "RSI Reversal",
            "Volatility Breakout"
        ]
    )

    initial_capital = st.sidebar.number_input("Initial Capital", value=1000000)

    ma_short = st.sidebar.slider("Short MA Length", 10, 100, 50)
    ma_long = st.sidebar.slider("Long MA Length", 100, 300, 200)
    deviation_threshold = st.sidebar.slider("Mean Reversion Deviation (%)", 1, 10, 3)
    rsi_period = st.sidebar.slider("RSI Period", 5, 30, 14)
    rsi_oversold = st.sidebar.slider("RSI Oversold Level", 10, 40, 30)

    # Indicators
    df["MA_Short"] = df[price_col].rolling(ma_short).mean()
    df["MA_Long"] = df[price_col].rolling(ma_long).mean()

    delta = df[price_col].diff()
    gain = delta.where(delta > 0, 0).rolling(rsi_period).mean()
    loss = -delta.where(delta < 0, 0).rolling(rsi_period).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    df["Volatility"] = df[price_col].rolling(20).std()

    df["Position"] = 0

    # Strategy Logic
    if strategy == "Buy & Hold":
        df["Position"] = 1

    elif strategy == "Momentum (Price > MA)":
        df["Position"] = np.where(df[price_col] > df["MA_Short"], 1, 0)

    elif strategy == "Dual MA Crossover":
        df["Position"] = np.where(df["MA_Short"] > df["MA_Long"], 1, 0)

    elif strategy == "Mean Reversion":
        deviation = (df[price_col] - df["MA_Short"]) / df["MA_Short"]
        df["Position"] = np.where(deviation < -deviation_threshold/100, 1, 0)

    elif strategy == "RSI Reversal":
        df["Position"] = np.where(df["RSI"] < rsi_oversold, 1, 0)

    elif strategy == "Volatility Breakout":
        breakout = df[price_col] > df[price_col].rolling(20).max().shift(1)
        df["Position"] = np.where(breakout, 1, 0)

    # Returns
    df["Return"] = df[price_col].pct_change()
    df["Strategy_Return"] = df["Return"] * df["Position"].shift(1)
    df["Equity"] = (1 + df["Strategy_Return"].fillna(0)).cumprod() * initial_capital

    # Buy & Hold Equity for comparison
    df["BH_Equity"] = (1 + df["Return"].fillna(0)).cumprod() * initial_capital

    # Metrics
    total_return = (df["Equity"].iloc[-1] / initial_capital - 1) * 100
    rolling_max = df["Equity"].cummax()
    drawdown = (df["Equity"] - rolling_max) / rolling_max
    max_dd = drawdown.min() * 100

    col1, col2 = st.columns(2)
    col1.metric("Strategy Return (%)", f"{total_return:.2f}")
    col2.metric("Max Drawdown (%)", f"{max_dd:.2f}")

    # Price Chart with Entries
    fig1, ax1 = plt.subplots()
    ax1.plot(df[date_col], df[price_col], label="Nifty")
    ax1.plot(df[date_col], df["MA_Short"], label="Short MA")

    entries = df[(df["Position"] == 1) & (df["Position"].shift(1) == 0)]
    exits = df[(df["Position"] == 0) & (df["Position"].shift(1) == 1)]

    ax1.scatter(entries[date_col], entries[price_col], marker="^")
    ax1.scatter(exits[date_col], exits[price_col], marker="v")

    ax1.legend()
    st.pyplot(fig1)

    # Equity Curve
    fig2, ax2 = plt.subplots()
    ax2.plot(df[date_col], df["Equity"], label="Strategy")
    ax2.plot(df[date_col], df["BH_Equity"], label="Buy & Hold")
    ax2.legend()
    st.pyplot(fig2)

    # Download performance
    st.download_button(
        "Download Strategy Data",
        df.to_csv(index=False),
        file_name="strategy_output.csv"
    )

else:
    st.info("Upload Nifty CSV to begin.")
