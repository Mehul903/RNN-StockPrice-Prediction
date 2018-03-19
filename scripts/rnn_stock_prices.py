"""
Created on Sun Mar 18 11:17:53 2018

@author: Mehul
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

## Load data
data_train = pd.read_csv('../dataset/Google_Stock_Price_Train.csv')
train = data_train.iloc[:,1:2].values

## Feature scaling
mms = MinMaxScaler(feature_range = (0,1))
train_scaled = mms.fit_transform(train)

## Create data structure
x_train = []
y_train = []

for stock in range(60, 1258):
    x_train.append(train_scaled[stock-60:stock, 0])
    y_train.append(train_scaled[stock,0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))


## Import modules for RNN
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

## Start building an RNN
regressor = Sequential()

## Adding first LSTM layer and dropout regularization:
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (x_train.shape[1], 1)))
regressor.add(Dropout(0.2))

## Adding second LSTM layer and dropout regularization:
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

## Adding third LSTM layer and dropout regularization:
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

## Adding fourth LSTM layer and dropout regularization:
regressor.add(LSTM(units = 50, return_sequences = False))
regressor.add(Dropout(0.2))

## Adding o/p layer:
regressor.add(Dense(units = 1))

## Compiling RNN:
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

## Fitting RNN to training set:
regressor.fit(x = x_train, y = y_train, epochs = 100, batch_size = 32)


## Making predictions and comparing results:
## Import test data:
data_test = pd.read_csv('../dataset/Google_Stock_Price_Test.csv')
test = data_test.iloc[:, 1:2].values

## Data preparaton for predictions:
data_all = pd.concat((data_train['Open'], data_test['Open']), axis = 0)
inputs  = data_all[len(data_train) - len(data_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = mms.transform(inputs)

x_test = []
for stock in range(60, 80):
    x_test.append(inputs[stock-60:stock, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

pred = regressor.predict(x_test)
pred = mms.inverse_transform(pred)

fig, ax = plt.subplots(figsize = (14,6))
ax.plot(test, color = 'red', label = 'True')
ax.plot(pred, color = 'k', label = 'Predictions')
ax.set_xlabel('Time', fontsize = 15)
ax.set_ylabel('Stock Price', fontsize = 15)
ax.legend()
plt.show()


