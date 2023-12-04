import pandas as pd
import numpy as np
import keras
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import joblib

class RNNCell:
    def __init__(self, model) -> None:
        self.model = model
        self._data = pd.read_csv('./data.csv',  parse_dates=['date'], index_col='date')
        self._predict = pd.DataFrame()
        self.__scaler = None


    def __create_data(self, type_fill = 'W'):
        dataset = self._data.copy()
        dataset = dataset.resample(type_fill).ffill()

        train_size = int(len(dataset) * 0.8)

        # train_data = df.WC.loc[:train_size] -----> it gives a series
        # Do not forget use iloc to select a number of rows
        train_data = dataset.iloc[:train_size]
        test_data = dataset.iloc[train_size:]

        # Scale data
        # The input to scaler.fit -> array-like, sparse matrix, dataframe of shape (n_samples, n_features)
        self.__scaler = MinMaxScaler().fit(train_data)

        train_scaled = self.__scaler.transform(train_data)
        test_scaled = self.__scaler.transform(test_data)   

        return self.__create_time_series_data(test_scaled, 48)



    def __create_time_series_data(self, X, look_back = 48):
        Xs, ys = [], []
        
        for i in range(len(X)-look_back):
            v = X[i:i+look_back]
            Xs.append(v)
            ys.append(X[i+look_back])
            
        return np.array(Xs), np.array(ys)


    def __predict(self, forecast_num, model,data,look_back):

        prediction_list = data[-look_back:]

        for _ in range(forecast_num):
            x = prediction_list[-look_back:]
        
            x = x.reshape((1, look_back, 1))
            out = model.predict(x)[0][0]
            prediction_list = np.append(prediction_list, out) 
        prediction_list = prediction_list[look_back-1:]

        return  self.__scaler.inverse_transform(prediction_list.reshape(-1,1))
    


    def get_result_df(self, model, days = 12, freq='W'):
        predict = self.get_result(model)
     
        last_date = self._data.index[-1]
        future_dates = pd.date_range(start=last_date , periods=len(predict), freq=freq)
        print(future_dates)
        self.__predict = pd.DataFrame(index= future_dates, columns=['price'])

        for i, price in zip(range(len(self.__predict)), predict):
            self.__predict.iloc[i] = price
        
        df_result = pd.concat([self._data, self.__predict], axis=0)
        return df_result
    


    def get_result(self, model):
        model_name = "model_" + model.lower() + ".keras"
        reconstructed_model = keras.models.load_model(model_name)
        X_test , _ = self.__create_data()
        return self.__predict(12, reconstructed_model, X_test[-1:], look_back=48)