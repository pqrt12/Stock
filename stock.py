#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/pqrt12/Stock/blob/master/stock.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Description: artificial recurrent neural network, Long Short Term Memory (LSTM).
#   Using 60 days stock price to predict the closing stock price of Apple.

# import
import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

# get the stock quote
df = web.DataReader('AAPL', data_source='yahoo', start='2012-01-01', end='2019-12-17')
df

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


# create the training dataset
train_data = scaled_data[0:training_data_len, :]

# split into x_train and y_train
x_train = []
y_train = []
for i in range(60, len(train_data)):
    x_train.append(train_data[i - 60 : i, 0])
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
test_data = scaled_data[training_data_len - 60 : , :]
x_test = []
y_test = dataset[training_data_len : , :]
for i in range(60, len(test_data)):
    x_test.append(test_data[i - 60 : i, 0])

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
valid = data[training_data_len : ]
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


size = 60
apple_quote = web.DataReader('AAPL', data_source='yahoo', start='2019-01-01', end='2020-05-14')
new_df = apple_quote.filter(['Close'])
last_60_days = new_df[-size : ].values
last_60_days_scaled = scaler.transform(last_60_days)
X_test = []
X_test.append(last_60_days_scaled)
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
pred_price = model.predict(X_test)
pred_price = scaler.inverse_transform(pred_price)
print(pred_price)

apple_quote2 = web.DataReader('AAPL', data_source='yahoo', start='2020-05-15', end='2020-05-15')
print(apple_quote2['Close'])

