import streamlit as st
import pandas as pd
import numpy as np
import keras
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import joblib

from model.RNNCell import RNNCell
from model.SLCell import SLCell

st.set_page_config(layout="wide")

df = pd.read_csv('./data.csv',  parse_dates=['date'], index_col='date')

dataset = df.copy()
dataset = dataset.resample('W').ffill()

# Using object notation
add_selectbox = st.sidebar.selectbox(
    "Lựa chọn loại hàng hóa?",
   ("Lúa gạo", "Giá xăng"),
   index=0,
   placeholder="Select contact method...",
)


optionModel = st.sidebar.selectbox(
    "Lựa chọn thuật toán?",
   ("LSTM", "GRU", "SVM", "XGBoost", "Random Forest"),
   index=0,
   placeholder="Select contact method...",
   key="2"
)



if(optionModel in ['LSTM', 'GRU']):
    prediction_gru = RNNCell(optionModel).get_result(optionModel)
else:
    prediction_gru = SLCell(optionModel).get_result(optionModel)

predicted_dates = pd.date_range(end='1/1/2023',freq ='W', periods=len(prediction_gru))


predicted_df = pd.DataFrame({'price_pred': prediction_gru}, index=predicted_dates)

merged_df = pd.concat([df, predicted_df], axis=0)

# Drop the redundant columns
# merged_df.drop(['price_x'], axis=1, inplace=True)


col1, col2 = st.columns([3, 1])

col1.title('Giá lúa sau dự đoán với mô hình ' + optionModel)
col1.line_chart(merged_df)
st.dataframe(merged_df)

col2.title('Giá lúa dự đoán')
col2.dataframe(predicted_df)
