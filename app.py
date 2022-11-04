import streamlit as st
import numpy as np
import keras

MODEL_PATH = "./model.h5"

st.title("Floating Population Warning")
# model = keras.models.load_model(MODEL_PATH)

_predict_option = st.selectbox("Select predict term", ("Long term predict", "Short term predict"))


if _predict_option == "Long term predict":
    st.header("Long term floating population predict using prophet")
elif _predict_option == "Short term predict":
    st.header("Short term floating population predict using GRU")

    ago_7 = float(st.number_input("7시간전 인구수"))
    ago_6 = float(st.number_input("6시간전 인구수"))
    ago_5 = float(st.number_input("5시간전 인구수"))
    ago_4 = float(st.number_input("4시간전 인구수"))
    ago_3 = float(st.number_input("3시간전 인구수"))
    ago_2 = float(st.number_input("2시간전 인구수"))
    ago_1 = float(st.number_input("1시간전 인구수"))

    def predict():
        row = np.array([[ago_7, ago_6]])

    