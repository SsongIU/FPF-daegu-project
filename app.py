import streamlit as st
import numpy as np
import pandas as pd
import prophet
import keras

MODEL_PATH = "./model.h5"
DATA_PATH = "./data/data.csv"

data = pd.read_csv(DATA_PATH)


def prophet_model(data, params):
    data_c = data.copy()
    data_c = data_c.rename(columns={"values": "y", "date": "ds"})
    m = prophet.Prophet(**params)
    m.add_seasonality(name="monthly", period=30.5, fourier_order=3, prior_scale=0.01, mode="multiplicative")
    m.fit(data_c)
    return m


# model = keras.models.load_model(MODEL_PATH)

st.title("Floating Population Warning")

_predict_option = st.selectbox("Select predict term", ("Long term predict", "Short term predict"))

if _predict_option == "Long term predict":
    st.header("Long term floating population predict using prophet")
    period = int(st.number_input("예측 기간"))

    params = {
    "changepoint_prior_scale": 0.5,
    "changepoint_range": 0.8,
    "seasonality_prior_scale": 10,
    "weekly_seasonality": True,
    "yearly_seasonality": False,
    "daily_seasonality": True,
    "seasonality_mode": "multiplicative",
    "holidays_prior_scale": 10,
    "interval_width": 0.8,
}
    model_prophet = prophet_model(data, params)

    future = model_prophet.make_future_dataframe(periods=period, freq='H')
    forecast = model_prophet.predict(future)
    # plot forecast
    fig = model_prophet.plot(forecast)
    a = prophet.plot.add_changepoints_to_plot(fig.gca(), model_prophet, forecast)
    # plot components
    fig2 = model_prophet.plot_components(forecast)
    st.write(fig)
    st.write(fig2)
elif _predict_option == "Short term predict":
    st.header("Short term floating population predict using GRU")

    ago_7 = float(st.number_input("7시간전 인구수"))
    ago_6 = float(st.number_input("6시간전 인구수"))
    ago_5 = float(st.number_input("5시간전 인구수"))
    ago_4 = float(st.number_input("4시간전 인구수"))
    ago_3 = float(st.number_input("3시간전 인구수"))
    ago_2 = float(st.number_input("2시간전 인구수"))
    ago_1 = float(st.number_input("1시간전 인구수"))

    # def predict():
    #     row = np.array([[ago_7, ago_6]])

    