import pandas as pd
import numpy as np
import keras
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import joblib

class SLCell:

    def __init__(self, model) -> None:
        self.model = model
        self._data = pd.read_csv('./data.csv',  parse_dates=['date'], index_col='date')
        self.__scaler = None


    def __create_data(self):
        dataset = self._data.copy()
        dataset = dataset.resample('W').ffill()

        train_size = int(len(dataset)*0.8)

        # train_data = df.WC.loc[:train_size] -----> it gives a series
        # Do not forget use iloc to select a number of rows
        train_data = dataset.iloc[:train_size]
        test_data = dataset.iloc[train_size:]

        # Scale data
        # The input to scaler.fit -> array-like, sparse matrix, dataframe of shape (n_samples, n_features)
        self.__scaler = MinMaxScaler().fit(train_data)

        train_scaled = self.__scaler.transform(train_data)
        test_scaled = self.__scaler.transform(test_data)   

        return self.__create_timeseries_data(test_scaled, 48)



    def __create_timeseries_data(self, X, look_back = 48):
        Xs, ys = [], []
        
        for i in range(len(X)-look_back):
            v = X[i:i+look_back]
            Xs.append(v)
            ys.append(X[i+look_back])
            
        return np.squeeze(np.array(Xs), axis=-1), np.array(ys)


    def __predict(self, forecast_num, model,data,look_back):
        time = 49

        df_gasolinePrices_pre = self._data.copy()
        gasolinePrices_scaled = self.__scaler.transform(df_gasolinePrices_pre)
                # Lấy dữ liệu cuối cùng từ tập dữ liệu
        last_data = gasolinePrices_scaled[-time:]
        last_data = last_data.reshape(1, -1)[:, -(time-1):]
        prediction_list = data[-look_back:]

        predicted_prices = []

        for day in range(forecast_num):
            next_prediction = model.predict(last_data)
            last_data = np.append(last_data, next_prediction).reshape(1, -1)[:, 1:]
            predicted_price = self.__scaler.inverse_transform(next_prediction.reshape(-1, 1))
            predicted_prices.append(predicted_price[0, 0])

        return predicted_prices
    
    
    def get_result_df(self, model, days=12, freq='W'):
        predict = self.get_result(model)

        last_date = self._data.index[-1]
        future_dates = pd.date_range(start=last_date , periods=len(predict), freq=freq)
        print(future_dates)
        predicted_df = pd.DataFrame(index=future_dates, columns=['price'])

        for i, price in zip(range(len(predicted_df)), predict):
            predicted_df.iloc[i] = price

        df_result = pd.concat([self._data, predicted_df], axis=0)
        return df_result
    

    
    def get_result(self, model):
        model_name = "model_" + model.lower().replace(' ', '')
        reconstructed_model = joblib.load(model_name + '.joblib')
        X_test = self.__create_data()
        return self.__predict(12, reconstructed_model, X_test[-1:], look_back=48)