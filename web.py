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
   ("Lúa gạo", "Giá xăng"),
   index=0,
   placeholder="Select contact method...",
)


option_model = st.sidebar.selectbox(
    "Lựa chọn thuật toán?",
   ("LSTM", "GRU", "SVM", "XGBoost", "Random Forest"),
   index=0,
   placeholder="Select contact method...",
   key="2"
)


predict, prediction_gru =  Commodity(option_model, './data.csv').get_predict(option_model, 'AGRICULTURAL')


kpi1, kpi2, kpi3 = st.columns(3)

    # fill in those three columns with respective metrics or KPIs
kpi1.metric(
        label="Trung bình giá lúa",
        value=round(predict.mean())
)


kpi2.metric(
        label="Giá lúa lớn nhất",
        value=round(predict.max())
)

kpi3.metric(
        label="Giá lúa thấp nhất",
        value=round(predict.min()),
)



col1, col2 = st.columns([3, 1])
# Drop the redundant columns
# merged_df.drop(['price_x'], axis=1, inplace=True)
col1.title('Giá lúa sau dự đoán với mô hình ' + option_model)
    # create three columns


fig = go.Figure()
fig.add_trace(go.Scatter(x=prediction_gru.index, y=prediction_gru['price'], name="Giá cũ", mode="lines"))


st.line_chart(prediction_gru)
st.dataframe(predict)
