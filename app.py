import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Trading Strategy Lab", layout="wide")
st.title("📚 Trading Strategy Learning Lab")

uploaded_file = st.file_uploader("Upload CSV (Date + Close)", type=["csv"])

if uploaded_file:

    # ================= LOAD =================
    df = pd.read_csv(uploaded_file)

    df.columns = [c.strip() for c in df.columns]

    # detect columns robustly
    date_col = None
    price_col = None

    for c in df.columns:
        if "date" in c.lower():
            date_col = c
        if "close" in c.lower() or "price" in c.lower():
            price_col = c

    if date_col is None or price_col is None:
        st.error("CSV must contain Date and Close/Price column")
        st.stop()

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[price_col] = df[price_col].astype(str).str.replace(",", "")
    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")

    df = df.dropna().sort_values(date_col).reset_index(drop=True)

    if len(df) < 50:
        st.warning("Very short dataset — some strategies may not trigger.")

    # ================= SIDEBAR =================
    strategy = st.sidebar.selectbox(
        "Strategy",
        [
            "Buy & Hold",
            "Momentum",
            "Dual MA",
            "Mean Reversion",
            "RSI",
            "Breakout",
            "Trend Pullback",
            "Blended"
        ]
    )

    ma_short = st.sidebar.slider("Short MA", 10, 80, 50)
    ma_long = st.sidebar.slider("Long MA", 100, 250, 200)
    rsi_period = st.sidebar.slider("RSI Period", 5, 30, 14)
    rsi_level = st.sidebar.slider("RSI Oversold", 10, 40, 30)
    deviation = st.sidebar.slider("Reversion %", 1, 10, 3)

    # ================= INDICATORS =================
    df["MA_S"] = df[price_col].rolling(ma_short).mean()
    df["MA_L"] = df[price_col].rolling(ma_long).mean()

    delta = df[price_col].diff()
    gain = delta.clip(lower=0).rolling(rsi_period).mean()
    loss = -delta.clip(upper=0).rolling(rsi_period).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    df["Return"] = df[price_col].pct_change()

    # ================= STRATEGIES =================
    def buy_hold():
        return pd.Series(1, index=df.index)

    def momentum():
        return (df[price_col] > df["MA_S"]).astype(int)

    def dual():
        return (df["MA_S"] > df["MA_L"]).astype(int)

    def reversion():
        dev = (df[price_col] - df["MA_S"]) / df["MA_S"]
        return (dev < -deviation/100).astype(int)

    def rsi():
        return (df["RSI"] < rsi_level).astype(int)

    def breakout():
        high = df[price_col].rolling(20).max().shift(1)
        return (df[price_col] > high).astype(int)

    def pullback():
        up = df["MA_S"] > df["MA_L"]
        pb = df[price_col] < df["MA_S"]
        return (up & pb).astype(int)

    def blended():
        return (
            momentum() +
            dual() +
            reversion() +
            rsi() +
            breakout()
        ) / 5

    strategy_map = {
        "Buy & Hold": buy_hold(),
        "Momentum": momentum(),
        "Dual MA": dual(),
        "Mean Reversion": reversion(),
        "RSI": rsi(),
        "Breakout": breakout(),
        "Trend Pullback": pullback(),
        "Blended": blended()
    }

    df["Position"] = strategy_map[strategy]
    df["Strat_Return"] = df["Return"] * df["Position"].shift(1)
    df["Equity"] = (1 + df["Strat_Return"].fillna(0)).cumprod()

    # ================= LEARNING PANELS =================

    st.subheader("📖 Strategy Explanation")
    
    explanations = {
    
    "Buy & Hold": """
    **Core idea:** Always stay invested.
    
    **When it works**
    • Long-term bull markets  
    • Strong economic growth  
    • Index investing  
    
    **When it fails**
    • Deep crashes  
    • Long sideways markets  
    
    **What to notice on chart**
    • Big drawdowns  
    • Strong long-term compounding  
    """,
    
    "Momentum": """
    **Core idea:** Buy strength (price above moving average).
    
    **When it works**
    • Strong trends  
    • Breakout markets  
    • Bull phases  
    
    **When it fails**
    • Sideways markets  
    • Choppy periods  
    • Frequent reversals  
    
    **What to notice**
    • Many small losses  
    • Few big winners  
    • Whipsaws near MA  
    """,
    
    "Dual MA": """
    **Core idea:** Confirm trend using two moving averages.
    
    **When it works**
    • Sustained bull markets  
    • Long macro trends  
    • Index trends  
    
    **When it fails**
    • Late entry  
    • Misses early rally  
    • Choppy markets  
    
    **What to notice**
    • Fewer trades  
    • Smoother curve  
    • Late entries  
    """,
    
    "Mean Reversion": """
    **Core idea:** Buy sharp dips expecting bounce.
    
    **When it works**
    • Range markets  
    • Low volatility  
    • Mean-reverting stocks  
    
    **When it fails**
    • Crashes  
    • Structural declines  
    • Strong trends  
    
    **What to notice**
    • Many quick trades  
    • Works until crash  
    • Big losses in trends  
    """,
    
    "RSI": """
    **Core idea:** Buy when market oversold.
    
    **When it works**
    • Short-term volatility  
    • Range markets  
    • Quick reversals  
    
    **When it fails**
    • Strong downtrends  
    • Bear markets  
    • Trending crashes  
    
    **What to notice**
    • Many signals  
    • False bottoms  
    • Overtrading  
    """,
    
    "Breakout": """
    **Core idea:** Buy new highs.
    
    **When it works**
    • Strong momentum  
    • News-driven rallies  
    • Trending stocks  
    
    **When it fails**
    • Fake breakouts  
    • Range markets  
    • Low volume  
    
    **What to notice**
    • Few trades  
    • Big winners  
    • Missed early move  
    """,
    
    "Trend Pullback": """
    **Core idea:** Buy dips within trend.
    
    **When it works**
    • Strong uptrends  
    • Healthy corrections  
    • Institutional buying  
    
    **When it fails**
    • Trend reversals  
    • Bear markets  
    • Fake pullbacks  
    
    **What to notice**
    • Better entries  
    • Lower drawdown  
    • Missed early breakout  
    """,
    
    "Blended": """
    **Core idea:** Combine multiple strategies.
    
    **When it works**
    • Mixed market regimes  
    • Diversification  
    • Long-term allocation  
    
    **When it fails**
    • Extreme trending markets  
    • Over-diversification  
    
    **What to notice**
    • Smoother equity  
    • Lower drawdown  
    • Moderate return  
    """
    }
    
    st.info(explanations[strategy])
    
    st.subheader("🎓 Classroom Learning Prompts")
    
    prompts = {
    "Buy & Hold": "Would an investor tolerate this drawdown?",
    "Momentum": "Why so many small losses?",
    "Dual MA": "Why is entry late?",
    "Mean Reversion": "Why does this crash fail?",
    "RSI": "Is oversold always a buy?",
    "Breakout": "Why so few trades?",
    "Trend Pullback": "Why better risk-reward?",
    "Blended": "Why smoother equity curve?"
    }
    
    st.write(prompts[strategy])

    # ================= CHART =================
    st.subheader("Chart")

    entries = df[(df["Position"]>0) & (df["Position"].shift(1)==0)]
    exits = df[(df["Position"]==0) & (df["Position"].shift(1)>0)]

    if strategy == "RSI":
        fig, (ax1, ax2) = plt.subplots(2,1,figsize=(10,7),sharex=True)

        ax1.plot(df[date_col], df[price_col])
        ax1.scatter(entries[date_col], entries[price_col], marker="^")
        ax1.scatter(exits[date_col], exits[price_col], marker="v")

        ax2.plot(df[date_col], df["RSI"])
        ax2.axhline(rsi_level, linestyle="--")
        ax2.axhline(70, linestyle="--")

        st.pyplot(fig)

    else:
        fig, ax = plt.subplots(figsize=(10,5))
        ax.plot(df[date_col], df[price_col])
        ax.plot(df[date_col], df["MA_S"])

        if strategy in ["Dual MA","Trend Pullback"]:
            ax.plot(df[date_col], df["MA_L"])

        if strategy=="Breakout":
            ax.plot(df[date_col], df[price_col].rolling(20).max(), linestyle="--")

        ax.scatter(entries[date_col], entries[price_col], marker="^")
        ax.scatter(exits[date_col], exits[price_col], marker="v")
        st.pyplot(fig)

    # ================= METRICS =================
    st.subheader("Metrics")

    if df["Equity"].notna().sum() > 0:
        ret = (df["Equity"].iloc[-1]-1)*100
        dd = (df["Equity"]/df["Equity"].cummax()-1).min()*100
        sharpe = df["Strat_Return"].mean()/df["Strat_Return"].std()*np.sqrt(252) if df["Strat_Return"].std()!=0 else 0
        trades = (df["Position"].diff()>0).sum()

        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Return %", f"{ret:.2f}")
        c2.metric("Sharpe", f"{sharpe:.2f}")
        c3.metric("MaxDD %", f"{dd:.2f}")
        c4.metric("Trades", int(trades))
    else:
        st.warning("No trades generated.")

    # ================= COMPARISON =================
    st.subheader("Strategy Comparison")

    results=[]
    for name,pos in strategy_map.items():
        r = df["Return"]*pos.shift(1)
        eq=(1+r.fillna(0)).cumprod()

        if eq.notna().sum()==0:
            results.append([name,0,0,0])
            continue

        ret=(eq.iloc[-1]-1)*100
        dd=(eq/eq.cummax()-1).min()*100
        sharpe=r.mean()/r.std()*np.sqrt(252) if r.std()!=0 else 0
        results.append([name,round(ret,2),round(sharpe,2),round(dd,2)])

    comp=pd.DataFrame(results,columns=["Strategy","Return%","Sharpe","MaxDD%"])
    st.dataframe(comp)

    # ================= EQUITY =================
    st.subheader("Equity Curve Comparison")

    fig2,ax2=plt.subplots(figsize=(10,5))
    for name,pos in strategy_map.items():
        r=df["Return"]*pos.shift(1)
        eq=(1+r.fillna(0)).cumprod()
        ax2.plot(df[date_col],eq,label=name)

    ax2.legend()
    st.pyplot(fig2)

else:
    st.info("Upload CSV to start.")
