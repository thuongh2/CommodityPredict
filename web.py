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
freq = 'W'
data_name = './data.csv'
if add_selectbox == 'Giá xăng':
    type = Commodity.OIL
    data_name = './df_oil.csv'
    freq = '3D'

if type == Commodity.AGRICULTURAL:
    month_select = st.sidebar.select_slider(
        'Chọn khoảng thời gian dự đoán',
        options=['1 tháng', '2 tháng', '3 tháng'])
else:
    month_select = st.sidebar.select_slider(
        'Chọn khoảng thời gian dự đoán',
        options=['1 tuần', '2 tuần', '3 tuần', '4 tuần'])

global time
month = month_select.split(" ")[0]
time = int(month) * 4



com =  Commodity(option_model, data_name)
predict, prediction_gru, predict_df = com.get_predict(option_model, type, freq, time)

st.title(add_selectbox + ' sau dự đoán với mô hình ' + option_model + " trong " + month_select )

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

st.title("Bảng giá dự đoán")
st.dataframe(df_format, width= 1000)


def render_footer():
    for _ in range(3):
        st.write('\n')
    st.markdown(
        "<br><hr><center>Developed by Võ Hoài Thương - 20133012 và Huỳnh Hạo Nhị - 20133006</center><hr>",
        unsafe_allow_html=True)
    st.markdown("<style> footer {visibility: hidden;} </style>", unsafe_allow_html=True)
    # HTML và CSS để sử dụng ảnh nền

render_footer()