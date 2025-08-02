import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf

from datetime import datetime, timedelta
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

st.title("Prediccion del bitcoin")
st.sidebar.header("Datos de entrada")
file_name = st.sidebar.file_uploader("Sube un archivo CSV con datos de Bitcoin", type=["csv"])

@st.cache_data

def get_btc_data(start='2014-01-01', end='2024-12-31'):
    btc = yf.download("BTC-USD", start=start, end=end)
    btc.reset_index(inplace=True)
    btc.to_csv("data/btc_raw.csv", index=False)
    return btc
    
def load_data(file_name=None)-> pd.DataFrame:
    if file_name:
        df = pd.read_csv(file_name, parse_dates=["Date"])
    else:
        get_btc_data()
        df = pd.read_csv("data/input.csv", parse_dates=["Date"])
    df = df.set_index("Date")
    df = df[pd.to_numeric(df['Close'], errors='coerce').notna()]
    df['Close'] = df['Close'].astype(float)
    df["day"] = df.index.day
    df["month"] = df.index.month
    df["year"] = df.index.year
    df["dayofweek"] = df.index.dayofweek
    df['Low'] = df['Low'].astype(float)
    df['High'] = df['High'].astype(float)
    df['Open'] = df['Open'].astype(float)

    df["lag_1"] = df["Close"].shift(1)
    df["lag_7"] = df["Close"].shift(7)
    df["lag_30"] = df["Close"].shift(30)
    df["rolling_mean_7"] = df["Close"].shift(1).rolling(window=7).mean()
    df["rolling_std_7"] = df["Close"].shift(1).rolling(window=7).std()
    df["rolling_mean_30"] = df["Close"].shift(1).rolling(window=30).mean()
    df["rolling_std_30"] = df["Close"].shift(1).rolling(window=30).std()

    df['price_change'] = df['Close'] - df['lag_1']
    df['volatility_14'] = df['Close'].rolling(window=14).std()
    df['hl_spread'] = df['High'] - df['Low']
    df['close_open_diff'] = df['Close'] - df['Open']
    df.dropna()
    
    return df

df = load_data(file_name)
st.subheader("Precio de cierre de bitcoin")
fig, ax = plt.subplots(figsize=(14, 6))
df['Close'].plot(ax=ax)
plt.ylabel("USD")
plt.grid()
plt.show()

cols = [column for column in df.columns if column != "Close"]
X = df[cols]
y = df['Close']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

tscv = TimeSeriesSplit(n_splits=5)
rmse_scores, mae_scores, mape_scores = [], [], []

y_pred_all = []
y_tests_all = []

for fold, (train_idx, test_idx) in enumerate(tscv.split(X_scaled)):
    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.01,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    y_pred_all.extend(y_pred)
    y_tests_all.extend(y_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

    rmse_scores.append(rmse)
    mae_scores.append(mae)
    mape_scores.append(mape)

    #print(f"Fold {fold+1}  RMSE: {rmse:.2f}, MAE: {mae:.2f}, MAPE: {mape:.2f}%")

#st.markdown("### Resultados Promedio:")
#st.write(f"RMSE: {np.mean(rmse_scores):.2f}")
#st.write(f"MAE: {np.mean(mae_scores):.2f}")
#st.write(f"MAPE: {np.mean(mape_scores):.2f}%")
    
final_model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.01,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

final_model.fit(X_scaled[:-30], y.iloc[:-30])
y_pred = final_model.predict(X_scaled[-30:])
y_true = y.iloc[-30:]

# Métricas
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae = mean_absolute_error(y_true, y_pred)
mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

st.markdown("### Métricas de Desempeño (últimos 30 días)")
st.write(f"**RMSE:** {rmse:.2f}")
st.write(f"**MAE:** {mae:.2f}")
st.write(f"**MAPE:** {mape:.2f}%")

st.subheader("Predicción vs Real (últimos 30 días)")
fig2, ax2 = plt.subplots(figsize=(12, 5))
ax2.plot(y_true.index, y_true, label="Real")
ax2.plot(y_true.index, y_pred, label="Predicción", alpha=0.8)
ax2.legend()
ax2.set_ylabel("USD")
ax2.grid()
st.pyplot(fig2)


st.subheader("Importancia de las Variables")
importances = pd.Series(final_model.feature_importances_, index=cols).sort_values()
fig3, ax3 = plt.subplots(figsize=(8, 6))
importances.plot(kind="barh", ax=ax3)
ax3.grid()
st.pyplot(fig3)

st.subheader("Configurar Predicción")
min_date = df.index[30]  # para asegurar que haya datos anteriores suficientes
max_date = df.index[-1]
default_date = max_date - timedelta(days=10)
selected_date = st.date_input("Selecciona la fecha de inicio de la predicción", min_value=min_date, max_value=max_date, value=default_date)

days_to_predict = st.slider("¿Cuántos días quieres predecir hacia adelante?", min_value=1, max_value=30, value=7)

try:
    start_idx = df.index.get_loc(pd.to_datetime(selected_date))
    X_pred = X_scaled[start_idx : start_idx + days_to_predict]
    y_real = y.iloc[start_idx : start_idx + days_to_predict]
    y_forecast = model.predict(X_pred)

    st.subheader(f"Predicción vs Real desde {selected_date} por {days_to_predict} días")
    fig4, ax4 = plt.subplots(figsize=(12, 5))
    ax4.plot(y_real.index, y_real, label="Real")
    ax4.plot(y_real.index, y_forecast, label="Predicción", alpha=0.8)
    ax4.legend()
    ax4.set_ylabel("USD")
    ax4.grid()
    st.pyplot(fig4)

    rmse = np.sqrt(mean_squared_error(y_real, y_forecast))
    mae = mean_absolute_error(y_real, y_forecast)
    mape = np.mean(np.abs((y_real - y_forecast) / y_real)) * 100

    st.markdown("### Métricas de Predicción Personalizada")
    st.write(f"**RMSE:** {rmse:.2f}")
    st.write(f"**MAE:** {mae:.2f}")
    st.write(f"**MAPE:** {mape:.2f}%")

except IndexError:
    st.warning("No hay suficientes datos disponibles después de esa fecha para predecir tantos días.")

