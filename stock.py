#!/usr/bin/env python
# coding: utf-8

# Description: artificial recurrent neural network, Long Short Term Memory (LSTM).
#   Using 60 days stock price to predict the closing stock price of Apple.

# import
import math
# import pandas_datareader as web
from yahoo_download import get_yahoo_hist_df, DATETIME_FORMAT
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')


df = get_yahoo_hist_df('AAPL', start_str='2012-01-01')
df['Date'] = pd.to_datetime(df['Date'], format=DATETIME_FORMAT)
df.head(3)

# visualize
plt.figure(figsize = (16, 8))
plt.title('Close Price History')
plt.plot(df['Close'])
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price', fontsize=18)
plt.show()

# create a dataframe with only the 'close' column
data = df.filter(['Close'])
# convert to numpy array
dataset = data.values
# the number of rows to train the model
training_data_len = math.ceil(len(dataset) * 0.8)
training_data_len

# scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)
scaled_data

size = 60

# create the training dataset
train_data = scaled_data[0:training_data_len, :]

# split into x_train and y_train
x_train = []
y_train = []
for i in range(size, len(train_data)):
    x_train.append(train_data[i - size : i, 0])
    y_train.append(train_data[i, 0])

# convert teh x_train and y_train to numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)
x_train.shape

# reshape the data
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_train.shape

# build the LSTM model 
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

# compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# train the model
# this takes a long long time.
model.fit(x_train, y_train, batch_size=1, epochs=1)

# testing dataset
test_data = scaled_data[training_data_len - size : , :]
x_test = []
y_test = dataset[training_data_len : , :]
for i in range(60, len(test_data)):
    x_test.append(test_data[i - size : i, 0])

# convert to numpy array
x_test = np.array(x_test)
# reshape
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# get the predicted values
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

# get the root mean square error (RMSE)
youtube = np.sqrt(np.mean(predictions - y_test) ** 2)
rmse = np.sqrt(np.mean((predictions - y_test) ** 2))
print(f'youtube={youtube}, rmse={rmse}')

# plot the data
train = data[: training_data_len]
valid = data[training_data_len : ].copy()
valid['Predictions'] = predictions
# Visualize
plt.figure(figsize=(16, 8))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price', fontsize = 18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()

