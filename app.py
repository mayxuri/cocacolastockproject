import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from finta import TA

st.set_page_config(page_title="Coca-Cola Stock Analysis", layout="wide")
st.title("🥤 Coca-Cola Stock Analysis and Live Prediction")

# --- 1. Load Data ---
# @st.cache_data
# def load_data():
#     hist = pd.read_csv("coca_cola_stock_history.csv")
#     info = pd.read_csv("coca_cola_stock_info.csv", header=None, names=["Description", "Information"])
#     hist['Date'] = pd.to_datetime(hist['Date'])
#     hist.sort_values('Date', inplace=True)
#     return hist, info

# data, info = load_data()
@st.cache_data
def load_data():
    try:
        hist = pd.read_csv("coca_cola_stock_history.csv")
        info = pd.read_csv("coca_cola_stock_info.csv", header=None, names=["Description", "Information"])
    except FileNotFoundError:
        st.stop()

    # Clean columns
    hist.columns = hist.columns.str.strip()

    # Parse date and remove timezone (convert to naive datetime)
    hist['Date'] = pd.to_datetime(hist['Date'], errors='coerce')
    hist['Date'] = hist['Date'].dt.tz_localize(None)

    # Drop any bad rows and sort
    hist.dropna(subset=['Date'], inplace=True)
    hist.sort_values('Date', inplace=True)

    return hist, info
data, info = load_data()

# --- 2. Show Info ---
with st.expander("📄 Basic Company Info"):
    st.dataframe(info.dropna())

# --- 3. EDA Charts ---
st.subheader("📈 Stock Price History")
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='Close'))
fig1.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='Open'))
fig1.update_layout(title='Coca-Cola Stock Closing vs Opening Price', xaxis_title='Date', yaxis_title='Price (USD)')
st.plotly_chart(fig1, use_container_width=True)

st.subheader("📊 Volume Over Time")
st.line_chart(data.set_index('Date')['Volume'])

# --- 4. Feature Engineering ---
data['MA_20'] = data['Close'].rolling(window=20).mean()
data['MA_50'] = data['Close'].rolling(window=50).mean()
data['Daily_Return'] = data['Close'].pct_change()
data['Volatility'] = data['Daily_Return'].rolling(window=20).std()
data.fillna(method='bfill', inplace=True)

# --- 5. Feature Selection ---
features = ['Open', 'High', 'Low', 'Volume', 'Dividends', 'Stock Splits', 'MA_20', 'MA_50', 'Daily_Return', 'Volatility']
target = 'Close'

X = data[features]
y = data[target]

# --- 6. Train-Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# --- 7. Train the Random Forest Model ---
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# --- 8. Predict and Evaluate ---
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

st.subheader("🤖 Model Evaluation")
st.write(f"**Mean Squared Error:** {mse:.2f}")
st.write(f"**Mean Absolute Error:** {mae:.2f}")

# Plot actual vs predicted
st.subheader("📉 Predicted vs Actual Closing Price")
compare_df = pd.DataFrame({"Date": data['Date'].iloc[-len(y_test):], "Actual": y_test, "Predicted": y_pred})
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=compare_df['Date'], y=compare_df['Actual'], name="Actual"))
fig2.add_trace(go.Scatter(x=compare_df['Date'], y=compare_df['Predicted'], name="Predicted"))
fig2.update_layout(title='Actual vs Predicted Closing Price', xaxis_title='Date', yaxis_title='Price (USD)')
st.plotly_chart(fig2, use_container_width=True)


# Show Moving Averages
st.subheader("📉 Moving Averages (MA20 vs MA50)")
st.line_chart(data.set_index('Date')[['Close', 'MA_20', 'MA_50']])

st.subheader("📡 Live Coca-Cola Stock Prediction (Today)")

# Get today's date
from datetime import datetime, timedelta
today = datetime.now().date()
start_live = today - timedelta(days=30)

# Download live stock data
live_data = yf.download('KO', start=start_live, end=today + timedelta(days=1), interval='1d', auto_adjust=False)
# live_data = live_data[live_data['Close'] > 0]  # Remove zero or invalid rows

# # Drop rows with missing OHLC data
# live_data.dropna(subset=['Open', 'High', 'Low', 'Close'], inplace=True)

if not live_data.empty:
    live_data.reset_index(inplace=True)
    live_data['MA_20'] = live_data['Close'].rolling(window=20).mean()
    live_data['MA_50'] = live_data['Close'].rolling(window=50).mean()
    live_data['Daily_Return'] = live_data['Close'].pct_change()
    live_data['Volatility'] = live_data['Daily_Return'].rolling(window=20).std()
    live_data.fillna(0, inplace=True)

    # Use the latest row for prediction
    latest = live_data.iloc[-1]
    latest_features = pd.DataFrame([{
        'Open': latest['Open'],
        'High': latest['High'],
        'Low': latest['Low'],
        'Volume': latest['Volume'],
        'Dividends': 0.0,
        'Stock Splits': 0.0,
        'MA_20': latest['MA_20'],
        'MA_50': latest['MA_50'],
        'Daily_Return': latest['Daily_Return'],
        'Volatility': latest['Volatility']
    }])

    # Make the prediction
    live_pred = model.predict(latest_features)[0]
    st.success(f"📌 **Predicted Closing Price for {today}: ${live_pred:.2f}**")

#     # Show today's chart
#     st.plotly_chart(go.Figure(
#         data=[go.Candlestick(
#             x=live_data['Date'],
#             open=live_data['Open'],
#             high=live_data['High'],
#             low=live_data['Low'],
#             close=live_data['Close']
#         )],
#         layout_title_text="Live Candlestick Chart (Last 30 Days)"
#     ), use_container_width=True)

# else:
#     st.warning("Live data could not be fetched. Please check your internet or try again later.")
#   # 📈 Line Chart of Closing Prices (Last 30 Days)
    st.subheader("📈 Live Closing Prices (Last 30 Days)")
    st.line_chart(live_data.set_index('Date')['Close'])

else:
    st.warning("⚠️ Unable to fetch live data. Please check your internet or try again later.")
