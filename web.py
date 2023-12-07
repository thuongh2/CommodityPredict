import streamlit as st
import pandas as pd
import numpy as np
import keras
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import joblib

import plotly.graph_objects as go

from model.RNNCell import RNNCell
from model.SLCell import SLCell
from model.Commodity import Commodity


# full screen
st.set_page_config(layout="wide")

# Using object notation
add_selectbox = st.sidebar.selectbox(
    "Lựa chọn loại hàng hóa?",
   ("Giá lúa", "Giá xăng"),
   index=0,
#    placeholder="Select contact method...",
)


option_model = st.sidebar.selectbox(
    "Lựa chọn thuật toán?",
   ("LSTM", "GRU", "SVM", "XGBoost", "Random Forest"),
   index=0,
#    placeholder="Select contact method...",
   key="2"
)

type = Commodity.AGRICULTURAL
data_name = './data.csv'
if add_selectbox == 'Giá xăng':
    type = Commodity.OIL
    data_name = './data_oil.csv'

com =  Commodity(option_model, data_name)
predict, prediction_gru, predict_df = com.get_predict(option_model, type)

st.title(add_selectbox + ' sau dự đoán với mô hình ' + option_model, )

kpi1, kpi2, kpi3 = st.columns(3)

    # fill in those three columns with respective metrics or KPIs
kpi1.metric(
        label= add_selectbox + " trung bình",
        value=round(predict.mean())
)


kpi2.metric(
        label= add_selectbox + " lớn nhất",
        value=round(predict.max())
)

kpi3.metric(
        label=add_selectbox +  " thấp nhất",
        value=round(predict.min()),
)



col1, col2 = st.columns([3, 1])

# tách màu giá lúa

st.line_chart(prediction_gru, color = ('#3440eb' , '#eb345f'))

df_format = predict_df
df_format.index = df_format.index.strftime('%d-%m-%Y')
st.dataframe(df_format, width= 500)
