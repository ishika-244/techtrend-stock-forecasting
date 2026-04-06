import streamlit as st
import pandas as pd
import numpy as np
import datetime
import plotly.graph_objs as go
import joblib

   
# # --- Layout Setup ---
st.set_page_config(page_title="📈 Stock Price Forecasting Dashboard", layout="wide")

# Background
st.markdown("""
<style>
.stApp {
    background: 
    linear-gradient(rgba(10,14,25,0.96), rgba(10,14,25,0.98)),
    url("https://images.unsplash.com/photo-1611974789855-9c2a0a7236a3?q=80&w=2070");

    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}
</style>
""", unsafe_allow_html=True)


# --- Sidebar ---

# st.sidebar.image("logo.png", width=120)
st.sidebar.title(" 📈 TechTrend")

st.sidebar.subheader("""
Smart Market Intelligence 
**Buy • Sell • Hold Signals**
""")


company = st.sidebar.selectbox(
    "🏢 Select Company",
    ["TCS", "Infosys", "Wipro", "HCLTech"]
)

model_choice = st.sidebar.selectbox("Choose Model", ["ARIMA", "XGBoost", "LSTM(IN Progress)"])
n_days = st.sidebar.slider("Number of Days to Forecast", 1, 30, 7)
st.sidebar.markdown("### 📊 View Options")
show_chart = st.sidebar.radio("Show Chart View", ["Estimated Prices", "Trend Line"])

st.sidebar.info("""
💡 Tip:
~ Short-term forecasts are more reliable.  
~ Use AI decision as guidance, not absolute truth.
""")

# --- Main Container ---

st.markdown(f"""
<div style="
padding: 25px;
border-radius: 16px;
background: rgba(255,255,255,0.04);
border: 1px solid rgba(255,255,255,0.08);
backdrop-filter: blur(12px);
margin-bottom:20px;
">

<h1 style="margin-bottom:5px;">TechTrend ~ AI Market Intelligence</h1>

<p style="color:#A0A7B8; font-size:14px;">
Real-time decision support using ML (XGBoost) + Time Series (ARIMA)
</p>

</div>
""", unsafe_allow_html=True)

# --- Display Selections ---

def card(content):
    st.markdown(f"""
    <div style="
    background: rgba(255,255,255,0.05);
    padding: 20px;
    border-radius: 15px;
    margin-bottom: 20px;
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.08);
    ">
    {content}
    </div>
    """, unsafe_allow_html=True)
     
st.subheader("Inputs")
col1, col2 = st.columns(2)

with col1:
    card(f"""
    <b>Company</b><br>{company}<br><br>
    <b>Model</b><br>{model_choice}
    """)

with col2:
    card(f"""
    <b>Forecast</b><br>{n_days} days<br><br>
    <b>View</b><br>{show_chart}
    """)


# Loading models

st.session_state["xgb_model"] =joblib.load(f"Models/xgb_model_{company}.joblib")
st.session_state["sc_xgb"] = joblib.load(f"Models/sc_xgb_{company}.joblib")

# Loading testing data
st.session_state["X_test_xgb"] = np.load(f"Processed/X_test_xgb_{company}.npy")

# XG Boost Prediction Function
def predict_with_xgb(n_days, X_test_xgb, xgb_model, sc_xgb):
    preds = xgb_model.predict(X_test_xgb[:n_days])
    return preds

# Arima Prediction Function 
from statsmodels.tsa.arima.model import ARIMA

def predict_with_ARIMA(df, n_days):
    series = df["Close"]
    model = ARIMA(series, order=(2,1,2))  
    model_fit = model.fit()

    forecast = model_fit.forecast(steps=n_days)
    return forecast.values

# Decision Making to buy , sell or hold the stocks for selected company
def make_decision(predictions, current_price):
    first_price = predictions[0]
    final_price = predictions[-1]
    
    market_change = (final_price - current_price) / current_price
    trend_change = (final_price - first_price) / first_price
    trend_change = max(min(trend_change, 0.2), -0.2)

    # DECISION LOGIC
    if market_change > 0.01:
         decision = "BUY 📈"
    elif market_change < -0.01:
         decision = "SELL 📉"
    else:
         decision = "HOLD 🤝"

    return decision, round(market_change * 100, 2), market_change, trend_change

df = pd.read_csv(f"data/raw/{company}.csv")
current_price = df["Close"].iloc[-1]

# --- Prediction Button ---
if st.button("⚡Run Analysis"):

    if "xgb_model" in st.session_state and "X_test_xgb" in st.session_state and "sc_xgb" in st.session_state:
            xgb_predictions = predict_with_xgb(
            n_days,
            st.session_state["X_test_xgb"],
            st.session_state["xgb_model"],
            st.session_state["sc_xgb"]
        )
    else:
            st.error("XGBoost model or data not loaded.")
            st.stop()

    arima_predictions = predict_with_ARIMA(df, n_days)

    if model_choice == "XGBoost":
         predicted_prices = xgb_predictions
    elif model_choice == "ARIMA":
         predicted_prices = arima_predictions
    elif model_choice == "LSTM(IN Progress)":
         st.warning("🚧 Model is in progress. Will be added shortly.")
         st.stop() 
    else:
            # fallback dummy
            predicted_prices = np.round(
            np.linspace(3600, 3600 + np.random.randint(20, 80), n_days), 2
            )
    
    future_dates = pd.date_range(start=datetime.datetime.today(), periods=n_days).strftime("%Y-%m-%d")
    df_pred = pd.DataFrame({"Date": future_dates, "Predicted Price": predicted_prices})
    
    st.success("Prediction generated successfully!")


    st.write("### 📈 Price Forecast")
    st.dataframe(df_pred, use_container_width=True)

    if show_chart == "Trend Line":
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_pred["Date"], y=df_pred["Predicted Price"], mode='lines+markers', name='Forecast'))
        fig.update_layout(title="Stock Price Trend Forecast", xaxis_title="Date", yaxis_title="Price", template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

    elif show_chart == "Estimated Prices":
        st.metric(label="📌 Last Day Estimate", value=f"{predicted_prices[-1]} ₹")
        st.metric(label="📈 First Day Estimate", value=f"{predicted_prices[0]} ₹")
    
    xgb_decision, _, xgb_market, xgb_trend = make_decision(xgb_predictions, current_price)
    arima_decision, _, arima_market, arima_trend = make_decision(arima_predictions, current_price)



    col1, col2 = st.columns([2,1])
    with col1:
         st.markdown("### 📈 Market Signals")
         st.markdown(f"""
                        <div style="
                         background: rgba(255,255,255,0.05);
                         padding: 20px;
                         border-radius: 14px;
                         ">
                         <b style="color:#4CAF50">XGBoost:</b> {xgb_decision} <br><br>
                         <b style="color:#FF9800">ARIMA:</b> {arima_decision}
                         </div>
                         """, unsafe_allow_html=True)

    with col2:
         st.markdown("### 📊 Change")
         st.markdown(f"""
                         <div style="
                         background: rgba(255,255,255,0.05);
                         padding: 20px;
                         border-radius: 14px;
                         ">
                         XGB: <b style="color:#00E5FF">{xgb_market*100:.2f}%</b><br>
                        ARIMA: <b style="color:#FFD54F">{arima_market*100:.2f}%</b>
                        </div>
                        """, unsafe_allow_html=True)

    
    # sanity check threshold (IMPORTANT)
    MAX_REASONABLE_CHANGE = 0.20   # 20%
    
    # ignore unrealistic predictions
    if abs(xgb_market) > MAX_REASONABLE_CHANGE:
         xgb_valid = False
    else:
         xgb_valid = True

    if abs(arima_market) > MAX_REASONABLE_CHANGE:
         arima_valid = False
    else:
         arima_valid = True

     # decision logic
    if xgb_valid and arima_valid:
         if xgb_decision == arima_decision:
              final_decision = xgb_decision
              confidence = "HIGH"

         else:
              if abs(xgb_market) > abs(arima_market):
                   final_decision = xgb_decision
                   confidence = "MEDIUM (XGB dominant)"
              else:
                   final_decision = arima_decision
                   confidence = "MEDIUM (ARIMA dominant)"

    elif xgb_valid:
         final_decision = xgb_decision
         confidence = "LOW (only XGB valid)"

    elif arima_valid:
         final_decision = arima_decision
         confidence = "LOW (only ARIMA valid)"

    else:
         final_decision = "HOLD 🤝"
         confidence = "VERY LOW (both unreliable)"

    decision_styles = {
             "BUY": "rgba(0, 200, 83, 0.15)",
             "SELL": "rgba(255, 82, 82, 0.15)",
             "HOLD": "rgba(255, 214, 0, 0.12)"
             }
    text_colors = {
             "BUY": "#00E676",
             "SELL": "#FF5252",
             "HOLD": "#FFD54F"
             }

    decision_type = final_decision.split()[0]

    bg = decision_styles.get(decision_type, "rgba(255,255,255,0.1)")
    text = text_colors.get(decision_type, "#fff")
    
    st.markdown(f"""
                <div style="
                background: {bg};
                backdrop-filter: blur(10px);
                padding: 24px;
                border-radius: 16px;
                text-align: center;
                margin-top: 25px;
                border: 1px solid rgba(255,255,255,0.08);
                ">
                <div style="font-size: 22px; font-weight: 600; color: {text};">
                {final_decision}
                </div>            
                <div style="font-size: 13px; color: #aaa; margin-top: 6px;">
                Confidence: {confidence}
                </div>
                </div>
                """, unsafe_allow_html=True)
    
    
    st.markdown("""
                <div style='
                background-color: rgba(255, 193, 7, 0.15);
                padding: 10px;
                border-radius: 8px;
                color: #FFD54F;
                '>
                ⚠️ XGBoost uses historical patterns, not true forecasting.
                </div>
                """, unsafe_allow_html=True)
# --- Footer ---
st.caption("⚠️ This is an experimental decision-support tool, not financial advice.")
st.caption("Built with ❤️ by Ishika")

