import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Trading Strategy Learning Lab", layout="wide")
st.title("📚 Trading Strategy Learning Lab")

st.write("Upload Nifty historical data and explore how different strategies behave.")

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

    # =====================
    # SIDEBAR
    # =====================
    st.sidebar.header("Strategy Controls")

    strategy = st.sidebar.selectbox(
        "Choose Strategy",
        ["Momentum (Single MA)", "Dual MA Crossover", "Mean Reversion", "RSI Reversal"]
    )

    ma_short = st.sidebar.slider("Short MA Length", 10, 100, 50)
    ma_long = st.sidebar.slider("Long MA Length", 100, 300, 200)
    deviation = st.sidebar.slider("Mean Reversion %", 1, 10, 3)
    rsi_period = st.sidebar.slider("RSI Period", 5, 30, 14)
    rsi_level = st.sidebar.slider("RSI Oversold", 10, 40, 30)

    # =====================
    # INDICATORS
    # =====================
    df["MA_S"] = df[price_col].rolling(ma_short).mean()
    df["MA_L"] = df[price_col].rolling(ma_long).mean()

    delta = df[price_col].diff()
    gain = delta.where(delta > 0, 0).rolling(rsi_period).mean()
    loss = -delta.where(delta < 0, 0).rolling(rsi_period).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    df["Position"] = 0

    # =====================
    # STRATEGY EXPLANATIONS
    # =====================

    if strategy == "Momentum (Single MA)":

        st.subheader("📖 Single Moving Average Strategy")

        st.markdown("""
        **Idea:** Buy when price is above moving average  
        **Works best:** strong trends  
        **Fails in:** sideways markets  

        Entry → Price crosses above MA  
        Exit → Price falls below MA
        """)

        df["Position"] = np.where(df[price_col] > df["MA_S"], 1, 0)

        st.info("Learning: Fast entries but many false signals.")

    elif strategy == "Dual MA Crossover":

        st.subheader("📖 Dual Moving Average Strategy")

        st.markdown("""
        **Idea:** Buy when short MA crosses above long MA  
        **Works best:** sustained bull markets  
        **Fails in:** choppy markets  

        Entry → Short MA > Long MA  
        Exit → Short MA < Long MA
        """)

        df["Position"] = np.where(df["MA_S"] > df["MA_L"], 1, 0)

        st.info("Learning: Fewer signals but enters late.")

    elif strategy == "Mean Reversion":

        st.subheader("📖 Mean Reversion Strategy")

        st.markdown("""
        **Idea:** Buy sharp dips expecting bounce  
        **Works best:** range markets  
        **Fails in:** crashes
        """)

        dev = (df[price_col] - df["MA_S"]) / df["MA_S"]
        df["Position"] = np.where(dev < -deviation/100, 1, 0)

        st.info("Learning: Buying dips works until a real crash happens.")

    elif strategy == "RSI Reversal":

        st.subheader("📖 RSI Reversal Strategy")

        st.markdown("""
        **Idea:** Buy when RSI oversold  
        **Works best:** volatile markets  
        **Fails in:** strong downtrends
        """)

        df["Position"] = np.where(df["RSI"] < rsi_level, 1, 0)

        st.info("Learning: Indicators generate many false signals.")

    # =====================
    # RETURNS
    # =====================
    df["Return"] = df[price_col].pct_change()
    df["Strat_Return"] = df["Return"] * df["Position"].shift(1)
    df["Equity"] = (1 + df["Strat_Return"].fillna(0)).cumprod()

    # =====================
    # ENTRY EXIT
    # =====================
    entries = df[(df["Position"]==1)&(df["Position"].shift(1)==0)]
    exits = df[(df["Position"]==0)&(df["Position"].shift(1)==1)]

    # =====================
    # CHART
    # =====================
    fig, ax = plt.subplots(figsize=(10,5))

    ax.plot(df[date_col], df[price_col], label="Price", linewidth=1.5)
    ax.plot(df[date_col], df["MA_S"], label="Short MA")

    if strategy == "Dual MA Crossover":
        ax.plot(df[date_col], df["MA_L"], label="Long MA")

    ax.scatter(entries[date_col], entries[price_col], marker="^", label="Entry")
    ax.scatter(exits[date_col], exits[price_col], marker="v", label="Exit")

    ax.legend()
    ax.set_title("Strategy Chart with Entry & Exit")
    st.pyplot(fig)

    # =====================
    # METRICS
    # =====================
    total_return = (df["Equity"].iloc[-1] - 1) * 100
    st.metric("Strategy Return %", f"{total_return:.2f}")

    # =====================
    # LEARNING PANEL
    # =====================
    st.subheader("🎓 What students should observe")

    if strategy == "Momentum (Single MA)":
        st.write("- Many trades")
        st.write("- Early entry")
        st.write("- Whipsaws in sideways markets")

    elif strategy == "Dual MA Crossover":
        st.write("- Fewer trades")
        st.write("- Late entry")
        st.write("- Smoother trend capture")

    elif strategy == "Mean Reversion":
        st.write("- Works in range")
        st.write("- Fails in crashes")

    elif strategy == "RSI Reversal":
        st.write("- Frequent signals")
        st.write("- Many false entries")

    # =====================
    # MULTI STRATEGY TABLE
    # =====================
    st.subheader("📊 Strategy Comparison (Simple)")

    comparison = pd.DataFrame({
        "Strategy": ["Buy & Hold", "Momentum", "Dual MA"],
        "Concept": ["Always invested", "Price vs MA", "MA vs MA"]
    })

    st.table(comparison)

else:
    st.info("Upload Nifty CSV to begin learning.")
