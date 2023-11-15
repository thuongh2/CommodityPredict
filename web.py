import streamlit as st
import pandas as pd
import numpy as np
import keras
from sklearn.preprocessing import MinMaxScaler, StandardScaler

chart_data = pd.DataFrame(np.random.randn(20, 3), columns=["a", "b", "c"])


df = pd.read_csv("../wfp_food_prices_vnm.csv", 
                 parse_dates=['date'], index_col='date')
df['price'] = df.groupby(df.index)['price'].transform('mean')
df = df.sort_index()
df = df[df['price'] != 0]
df = df[df['admin1'] == 'An Giang']
df = df.drop_duplicates()
df = df.loc[:,['price']]

reconstructed_model = keras.models.load_model("model_gru.keras")

train_size = int(len(df)*0.8)

# train_data = df.WC.loc[:train_size] -----> it gives a series
# Do not forget use iloc to select a number of rows
train_data = df.iloc[:train_size]
test_data = df.iloc[train_size:]
new_data = test_data[-50:]
scaler = MinMaxScaler().fit(train_data)
test_scaled = scaler.transform(new_data)

# Reshape the input 
def create_dataset (X, look_back = 1):
    Xs = []
    for i in range(len(X)-look_back):
        v = X[i:i+look_back]
        Xs.append(v)
        
    return np.array(Xs)

X_30= create_dataset(test_scaled,12)

def prediction(model):
    prediction = model.predict(X_30)
    prediction = scaler.inverse_transform(prediction)
    return prediction

prediction_gru = prediction(reconstructed_model)
predicted_dates = pd.date_range(start=test_data.index[-1], freq ='M', periods=len(prediction_gru))


predicted_df = pd.DataFrame({'price': prediction_gru.flatten()}, index=predicted_dates)
merged_df = pd.merge(df, predicted_df, left_index=True, right_index=True, how='outer')
merged_df['price'] = merged_df[['price_x', 'price_y']].sum(axis=1)
# Drop the redundant columns
merged_df.drop(['price_x'], axis=1, inplace=True)

option = st.selectbox(
   "Lựa chọn thuật toán?",
   ("Email", "Home phone", "Mobile phone"),
   index=None,
   placeholder="Select contact method...",
)

st.title('Dữ liệu giá lúa')
st.line_chart(df)
st.title('Giá lúa sau dự đoán')
st.line_chart(merged_df,  color=["#FFBF00", "#C70039"])
