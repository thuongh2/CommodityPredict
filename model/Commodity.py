import pandas as pd
import numpy as np
import keras
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import joblib
import tensorflow as tf

class Commodity:

    AGRICULTURAL = 'AGRICULTURAL'
    OIL = 'OIL'

    def __init__(self, model, data) -> None:
        self.model = model
        self._data = pd.read_csv(data, parse_dates=['date'], index_col='date')
        self._predict = pd.DataFrame()
        self.__scaler = None

    
    def __create_data(self, mode, lool_back = 48, type_fill = 'W'):
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

        return self.__create_time_series_data(test_scaled, lool_back, mode)
    

    def __create_time_series_data(self, X, look_back = 48, mode = 'rnn'):
        Xs, ys = [], []
        
        for i in range(len(X) - look_back):
            v = X[i:i+look_back]
            Xs.append(v)
            ys.append(X[i+look_back])
        if(mode == 'rnn'):
            return np.array(Xs), np.array(ys)
        return np.squeeze(np.array(Xs), axis=-1), np.array(ys)
    


    def __predict_rnn(self, forecast_num, model, data, look_back=48):

        predicted_prices = data[-look_back:]

        for _ in range(forecast_num):
            x = predicted_prices[-look_back:]
        
            x = x.reshape((1, look_back, 1))
            out = model.predict(x)[0][0]
            predicted_prices = np.append(predicted_prices, out) 
        predicted_prices = predicted_prices[look_back-1:]

        return  self.__scaler.inverse_transform(predicted_prices.reshape(-1,1))
    

    def __predict_ensemble(self, forecast_num, model, data, look_back, time = 49):

        dataset = self._data.copy()
        dataset = self.__scaler.transform(dataset)
        # Lấy dữ liệu cuối cùng từ tập dữ liệu
        last_data = dataset[-time:]
        last_data = last_data.reshape(1, -1)[:, -(time-1):]

        predicted_prices = []

        for day in range(forecast_num):
            next_prediction = model.predict(last_data)
            last_data = np.append(last_data, next_prediction).reshape(1, -1)[:, 1:]
            predicted_price = self.__scaler.inverse_transform(next_prediction.reshape(-1, 1))
            predicted_prices.append(predicted_price[0, 0])

        return predicted_prices
    
    
    def get_result(self, model, type, time):

        assert type != None

        mode = 'rnn' if model in ['LSTM', 'GRU'] else 'en'

        global folder

        look_back = 48
        X_test = pd.DataFrame()
        if type == self.AGRICULTURAL:
            X_test, _ = self.__create_data(mode)
            folder =  'algricultural'
        else:
            look_back = 12
            X_test, _ = self.__create_data(mode, lool_back = look_back)
            folder = 'oil'

        # model_name = './' + folder + '/model_' + model.lower().replace(' ', '') + '.joblib'
      
        if model in ['LSTM', 'GRU']:
            model_name = './%s/model_%s.h5' % (folder,  model.lower().replace(' ', ''))
            reconstructed_model = tf.keras.models.load_model(model_name)
        else:
            model_name = './%s/model_%s.joblib' % (folder,  model.lower().replace(' ', ''))
            reconstructed_model = joblib.load(model_name)

        print(model_name)

    
        if model in ['LSTM', 'GRU']:
            self._predict = self.__predict_rnn(time, reconstructed_model, X_test[-1:], look_back= look_back)
        else:
            self._predict = self.__predict_ensemble(time, reconstructed_model, X_test[-1:], look_back= look_back, time= look_back + 1)
        return self._predict


    def get_result_df(self, predict_list, type, freq= 'W'):

        last_date = self._data.index[-1] 
        future_dates = pd.date_range(start=last_date , periods=len(predict_list), freq=freq)
        predicted_df = pd.DataFrame(index=future_dates, columns=['price_predict'])

        for i, price in zip(range(len(predicted_df)), predict_list):
            predicted_df.iloc[i] = price
      
        plot_time = int(len(self._data) * 0.6)
        if(type == self.OIL):
            df_oil = pd.read_csv('./data/gasoline.csv', parse_dates=['date'], index_col='date')
            self._data = df_oil
            plot_time = 0
        

        df_result = pd.concat([self._data[plot_time:], predicted_df], axis= 1)
        df_result.loc[predicted_df.index[0], 'price'] = predict_list[0]

        # df_result = self._data.join(predict_list)

        return df_result, predicted_df
    

    def get_predict(self, model, type, freq,  time= 12):
        self._predict = self.get_result(model, type, time)

        df_predict_full, predicted_df = self.get_result_df(predict_list= self._predict,  type=type, freq=freq)
    
        return np.array(self._predict), df_predict_full, predicted_df
    


        




