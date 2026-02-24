import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Trading Learning Lab", layout="wide")
st.title("📚 Trading Strategy Learning Lab")

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

    # Sidebar
    st.sidebar.header("Strategy Controls")

    strategy = st.sidebar.selectbox(
        "Choose Strategy",
        [
            "Momentum (MA)",
            "Dual MA",
            "Mean Reversion",
            "RSI Reversal"
        ]
    )

    ma_short = st.sidebar.slider("Short MA Length", 10, 100, 50)
    ma_long = st.sidebar.slider("Long MA Length", 100, 300, 200)
    deviation = st.sidebar.slider("Mean Reversion %", 1, 10, 3)
    rsi_period = st.sidebar.slider("RSI Period", 5, 30, 14)
    rsi_level = st.sidebar.slider("RSI Oversold", 10, 40, 30)

    # Indicators
    df["MA_S"] = df[price_col].rolling(ma_short).mean()
    df["MA_L"] = df[price_col].rolling(ma_long).mean()

    delta = df[price_col].diff()
    gain = delta.where(delta > 0, 0).rolling(rsi_period).mean()
    loss = -delta.where(delta < 0, 0).rolling(rsi_period).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    df["Position"] = 0

    # ===== STRATEGY LOGIC =====

    if strategy == "Momentum (MA)":
        st.subheader("📖 Strategy Explanation")
        st.write("""
        **Idea:** Buy when price is above moving average.  
        **Works in:** strong trends  
        **Fails in:** sideways markets (whipsaws)
        """)

        df["Position"] = np.where(df[price_col] > df["MA_S"], 1, 0)

        st.info("Learning: Trend-following enters late but captures big moves.")

    elif strategy == "Dual MA":
        st.subheader("📖 Strategy Explanation")
        st.write("""
        **Idea:** Buy when short MA crosses above long MA.  
        **Works in:** sustained bull markets  
        **Fails in:** choppy markets
        """)

        df["Position"] = np.where(df["MA_S"] > df["MA_L"], 1, 0)

        st.info("Learning: Confirmation reduces false signals but adds lag.")

    elif strategy == "Mean Reversion":
        st.subheader("📖 Strategy Explanation")
        st.write("""
        **Idea:** Buy sharp dips expecting bounce.  
        **Works in:** range markets  
        **Fails in:** crashes
        """)

        dev = (df[price_col] - df["MA_S"]) / df["MA_S"]
        df["Position"] = np.where(dev < -deviation/100, 1, 0)

        st.info("Learning: Buying dips works until it doesn't.")

    elif strategy == "RSI Reversal":
        st.subheader("📖 Strategy Explanation")
        st.write("""
        **Idea:** Buy when RSI is oversold.  
        **Works in:** volatile markets  
        **Fails in:** strong downtrends
        """)

        df["Position"] = np.where(df["RSI"] < rsi_level, 1, 0)

        st.info("Learning: Indicators give many false signals.")

    # ===== RETURNS =====
    df["Return"] = df[price_col].pct_change()
    df["Strat_Return"] = df["Return"] * df["Position"].shift(1)
    df["Equity"] = (1 + df["Strat_Return"].fillna(0)).cumprod()

    # ===== ENTRY EXIT =====
    entries = df[(df["Position"]==1)&(df["Position"].shift(1)==0)]
    exits = df[(df["Position"]==0)&(df["Position"].shift(1)==1)]

    # ===== CHART =====
    fig, ax = plt.subplots()
    ax.plot(df[date_col], df[price_col], label="Price")
    ax.plot(df[date_col], df["MA_S"], label="MA")

    ax.scatter(entries[date_col], entries[price_col], marker="^")
    ax.scatter(exits[date_col], exits[price_col], marker="v")

    ax.legend()
    st.pyplot(fig)

    # ===== METRICS =====
    total_return = (df["Equity"].iloc[-1] - 1) * 100
    st.metric("Return %", f"{total_return:.2f}")

    st.subheader("🎓 What Students Should Observe")
    st.write("""
    - Where does strategy enter?
    - Is entry late?
    - How many false signals?
    - What market regime helps it?
    """)

else:
    st.info("Upload Nifty CSV to start learning.")
