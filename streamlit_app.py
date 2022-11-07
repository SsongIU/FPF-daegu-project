import streamlit as st
import numpy as np
import pandas as pd
import prophet
import keras
import os
import matplotlib.pyplot as plt

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

MODEL_PATH = "./model/model_population.h5"
DATA_PATH = "./data/data.csv"

st.title("Floating Population Forecast")

_predict_option = st.selectbox("Select predict term", ("Long term predict", "Short term predict"))

if _predict_option == "Long term predict":
    st.header("Long term floating population predict using prophet")
    # prophet data
    data = pd.read_csv(DATA_PATH)

    # prophet model
    def prophet_model(data, params):
        data_c = data.copy()
        data_c = data_c.rename(columns={"values": "y", "date": "ds"})
        m = prophet.Prophet(**params)
        m.add_seasonality(name="monthly", period=30.5, fourier_order=3, prior_scale=0.01, mode="multiplicative")
        m.fit(data_c)
        return m

    col1, col2 = st.columns([3, 3])
    cps = float(col1.select_slider("changepoint_prior_scale", options=[0.005, 0.025, 0.05, 0.1, 0.2], value=0.05))
    sps = float(col2.select_slider("seasonality_prior_scale", options=[1, 10, 15, 20, 25, 30, 50], value=20))

    # predict period
    period = int(st.number_input("예측 기간 (시간)", step=1, max_value=5000))
    params = {
        "changepoint_prior_scale": cps,
        "changepoint_range": 0.8,
        "seasonality_prior_scale": sps,
        "weekly_seasonality": True,
        "yearly_seasonality": False,
        "daily_seasonality": True,
        "seasonality_mode": "multiplicative",
        "holidays_prior_scale": 10,
        "interval_width": 0.8,
    }
    model_prophet = prophet_model(data, params)

    future = model_prophet.make_future_dataframe(periods=period, freq="H")
    forecast = model_prophet.predict(future)
    # plot forecast
    fig = model_prophet.plot(forecast)
    a = prophet.plot.add_changepoints_to_plot(fig.gca(), model_prophet, forecast)
    # plot components
    fig2 = model_prophet.plot_components(forecast)
    st.write(fig)
    st.write(fig2)
elif _predict_option == "Short term predict":
    st.header("Short term floating population predict using LSTM")

    col1, col2 = st.columns([3, 3])

    model = keras.models.load_model(MODEL_PATH)

    ago_7 = int(col1.slider("6시간전 인구수", step=10, value=3000, max_value=30000))
    ago_6 = int(col1.slider("5시간전 인구수", step=10, value=3500, max_value=30000))
    ago_5 = int(col1.slider("4시간전 인구수", step=10, value=5500, max_value=30000))
    ago_4 = int(col1.slider("3시간전 인구수", step=10, value=6500, max_value=30000))
    ago_3 = int(col2.slider("2시간전 인구수", step=10, value=7500, max_value=30000))
    ago_2 = int(col2.slider("1시간전 인구수", step=10, value=10000, max_value=30000))
    ago_1 = int(col2.slider("지금 인구수", step=10, value=13000, max_value=30000))

    time_list = [ago_7, ago_6, ago_5, ago_4, ago_3, ago_2, ago_1]
    time_list_x = ["6h ago", "5h ago", "4h ago", "3h ago", "2h ago", "1h ago", "now"]
    pred_list = [ago_1]
    pred_list_x = ["now", "1h after"]
    x_ticks = ["6h ago", "5h ago", "4h ago", "3h ago", "2h ago", "1h ago", "now", "1h after"]

    pred_value = np.array([[ago_7, ago_6, ago_5, ago_4, ago_3, ago_2, ago_1]])

    if st.button("Predict"):
        pred = int(model.predict(pred_value))
        col2.subheader(f"1시간 후 예상인구: {pred}명")
        if pred > 15000:
            st.error("유동인구수가 기준치를 초과하였습니다!")

        pred_list.append(pred)
        fig, ax = plt.subplots()
        ax.plot(time_list_x, time_list)
        ax.plot(pred_list_x, pred_list, color="red")
        ax.set_xticks(x_ticks)
        st.pyplot(fig)
