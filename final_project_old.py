# Ryan Bell

import pandas as pd
from sklearn import preprocessing
import numpy as np

import keras
import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Dropout, LSTM, Input, Activation, concatenate
from keras import optimizers
import numpy as np
np.random.seed(4)
from tensorflow import set_random_seed
set_random_seed(4)

history_points = 50

def load_data(csv_path):
    data = pd.read_csv(csv_path)
    data = data.drop('Date', axis=1)
    data = data.drop('Adj Close', axis=1)
    data = data.drop(0, axis=0)
    data = data.values

    data_normaliser = preprocessing.MinMaxScaler()
    data_normalised = data_normaliser.fit_transform(data)

    # using the last {history_points} open close high low volume data points, predict the next open value
    ohlcv_histories_normalised = np.array(
        [data_normalised[i:i + history_points].copy() for i in range(len(data_normalised) - history_points)])
    next_day_open_values_normalised = np.array(
        [data_normalised[:, 0][i + history_points].copy() for i in range(len(data_normalised) - history_points)])
    next_day_open_values_normalised = np.expand_dims(next_day_open_values_normalised, -1)

    next_day_open_values = np.array([data[:, 0][i + history_points].copy() for i in range(len(data) - history_points)])
    next_day_open_values = np.expand_dims(next_day_open_values, -1)

    y_normaliser = preprocessing.MinMaxScaler()
    y_normaliser.fit(next_day_open_values)

    assert ohlcv_histories_normalised.shape[0] == next_day_open_values_normalised.shape[0]
    return ohlcv_histories_normalised, next_day_open_values_normalised, next_day_open_values, y_normaliser

def build_model():
    lstm_input = Input(shape=(history_points, 5), name='lstm_input')
    x = LSTM(50, name='lstm_0')(lstm_input)
    x = Dropout(0.2, name='lstm_dropout_0')(x)
    x = Dense(64, name='dense_0')(x)
    x = Activation('sigmoid', name='sigmoid_0')(x)
    x = Dense(1, name='dense_1')(x)
    output = Activation('linear', name='linear_output')(x)
    model = Model(inputs=lstm_input, outputs=output)

    adam = optimizers.Adam(lr=0.0005)

    model.compile(optimizer=adam, loss='mse')
    return model

def train_model(model, ohlcv_train, y_train, ohlcv_test, y_test):
    model.fit(x=ohlcv_train, y=y_train, batch_size=32, epochs=50, shuffle=True, validation_split=0.1)
    evaluation = model.evaluate(ohlcv_test, y_test)
    print(evaluation)

def main():
    APPL_ohlcv_histories, APPL_next_day_open_values, APPL_unscaled_y, APPL_y_normaliser = load_data('AAPL.csv')
    F_ohlcv_histories, F_next_day_open_values, F_unscaled_y, F_y_normaliser = load_data('F.csv')
    GM_ohlcv_histories, GM_next_day_open_values, GM_unscaled_y, GM_y_normaliser = load_data('GM.csv')
    MANT_ohlcv_histories, MANT_next_day_open_values, MANT_unscaled_y, MANT_y_normaliser = load_data('MANT.csv')
    MRNA_ohlcv_histories, MRNA_next_day_open_values, MRNA_unscaled_y, MRNA_y_normaliser = load_data('MRNA.csv')
    RDS_ohlcv_histories, RDS_next_day_open_values, RDS_unscaled_y, RDS_y_normaliser = load_data('RDS-B.csv')
    WDC_ohlcv_histories, WDC_next_day_open_values, WDC_unscaled_y, WDC_y_normaliser = load_data('WDC.csv')

    test_split = 0.9  # the percent of data to be used for testing
    n = []
    n.append(int(APPL_ohlcv_histories.shape[0] * test_split))
    n.append(int(F_ohlcv_histories.shape[0] * test_split))
    n.append(int(GM_ohlcv_histories.shape[0] * test_split))
    n.append(int(MANT_ohlcv_histories.shape[0] * test_split))
    n.append(int(MRNA_ohlcv_histories.shape[0] * test_split))
    n.append(int(RDS_ohlcv_histories.shape[0] * test_split))
    n.append(int(WDC_ohlcv_histories.shape[0] * test_split))

    ohlcv_train = []
    ohlcv_train.append(APPL_ohlcv_histories[:n[0]])
    ohlcv_train.append(F_ohlcv_histories[:n[1]])
    ohlcv_train.append(GM_ohlcv_histories[:n[2]])
    ohlcv_train.append(MANT_ohlcv_histories[:n[3]])
    ohlcv_train.append(MRNA_ohlcv_histories[:n[4]])
    ohlcv_train.append(RDS_ohlcv_histories[:n[5]])
    ohlcv_train.append(WDC_ohlcv_histories[:n[6]])

    y_train = []
    y_train.append(APPL_next_day_open_values[:n[0]])
    y_train.append(F_next_day_open_values[:n[1]])
    y_train.append(GM_next_day_open_values[:n[2]])
    y_train.append(MANT_next_day_open_values[:n[3]])
    y_train.append(MRNA_next_day_open_values[:n[4]])
    y_train.append(RDS_next_day_open_values[:n[5]])
    y_train.append(WDC_next_day_open_values[:n[6]])

    ohlcv_test = []
    ohlcv_test.append(APPL_ohlcv_histories[n[0]:])
    ohlcv_test.append(F_ohlcv_histories[n[1]:])
    ohlcv_test.append(GM_ohlcv_histories[n[2]:])
    ohlcv_test.append(MANT_ohlcv_histories[n[3]:])
    ohlcv_test.append(MRNA_ohlcv_histories[n[4]:])
    ohlcv_test.append(RDS_ohlcv_histories[n[5]:])
    ohlcv_test.append(WDC_ohlcv_histories[n[6]:])

    y_test = []
    y_test.append(APPL_next_day_open_values[n[0]:])
    y_test.append(F_next_day_open_values[n[1]:])
    y_test.append(GM_next_day_open_values[n[2]:])
    y_test.append(MANT_next_day_open_values[n[3]:])
    y_test.append(MRNA_next_day_open_values[n[4]:])
    y_test.append(RDS_next_day_open_values[n[5]:])
    y_test.append(WDC_next_day_open_values[n[6]:])

    unscaled_y_test = []
    unscaled_y_test.append(APPL_unscaled_y[n[0]:])
    unscaled_y_test.append(F_unscaled_y[n[1]:])
    unscaled_y_test.append(GM_unscaled_y[n[2]:])
    unscaled_y_test.append(MANT_unscaled_y[n[3]:])
    unscaled_y_test.append(MRNA_unscaled_y[n[4]:])
    unscaled_y_test.append(RDS_unscaled_y[n[5]:])
    unscaled_y_test.append(WDC_unscaled_y[n[6]:])

    models = []
    for i in range(len(n)):
        models.append(build_model())

    for i in range(len(n)):
        train_model(models[i], ohlcv_train[i], y_train[i], ohlcv_test[i], y_test[i])

main()