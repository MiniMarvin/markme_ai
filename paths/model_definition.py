# Must exist two networks one for the theft probability and one for the time prediction
#
# RNN base
# the network input must recieve the city name and the theft or not label
#
# CNN for risk prediction
# get as the input the entire route and give the probability of being stolen in this route
#
# CNN base
# the network input must recieve the entire path and them it must return the time prediction for the path
#
# Objective for better performance: Make a RNN to the time prediction where the variable input is the actual city 
# and the non variable input is the entire path

#############################################
## Import all libs
#############################################
# LSTM for international airline passengers problem with regression framing
import numpy
import matplotlib.pyplot as plt
from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

#############################################
## Helper function
#############################################
# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset) - look_back - 1):
		a = dataset[i:(i + look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return numpy.array(dataX), numpy.array(dataY)

# fix random seed for reproducibility
numpy.random.seed(7)

#############################################
## Define the dataset
#############################################
# load the dataset
# dataframe = read_csv('international-airline-passengers.csv', usecols=[1], engine='python', skipfooter=3)
dataframe = read_csv('series.csv', usecols=[1], engine='python', skipfooter=0)
dataset = dataframe.values
dataset = dataset.astype('float32')

# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

# dataset = numpy.array([[(2*x + 1)/(2*99+1)] for x in range(0, 100)])
# dataset = numpy.array([[(0.3)*(-1)**x] for x in range(0, 200)])

# split into train and test sets
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

# reshape into X=t and Y=t+1
look_back = 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

# reshape input to be [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

#############################################
## Define the model
#############################################
# create and fit the LSTM network
model = Sequential()
# model.add(LSTM(4, input_shape=(1, look_back), return_sequences=True))
model.add(LSTM(4, input_shape=(1, look_back), return_sequences=True))
model.add(LSTM(16, input_shape=(1, look_back), return_sequences=True))
model.add(LSTM(64, input_shape=(1, look_back), return_sequences=False))
model.add(Dropout(rate=0.3))
# model.add(Dense(3))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=1000, batch_size=5, verbose=2)

# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)


## Normalize the predictions
trainPredict = trainPredict/max(trainPredict)
testPredict = testPredict/max(testPredict)

# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))

model.save_weights("model-temporal-series.h5")
print("Saved model to disk")


# shift train predictions for plotting
trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

# shift test predictions for plotting
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict

# plot baseline and predictions
plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()