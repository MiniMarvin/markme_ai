#########################################################
## WARNING: this script takes 10 seconds to run in the
## dataset containing almost 60000 entries 
#########################################################

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def splitGroups(dataset, depth, maxDepth):
	"""
	A recursive function to split the list in two computing the ones that are next to
	the border of the set
	"""
	group1 = [dataset[0]] # max
	group2 = [dataset[len(dataset) - 1]] # min

	selector = [1, -1]

	## Goes computing the difference between the two groups and join to the list 
	## where the valueapproaches best
	for pivot in range(1, len(dataset)//2):
		for i in range(0, 2):
			if selector[i] == 1:
				data = max(dataset[pivot:len(dataset) - pivot])
			else:
				data = min(dataset[pivot:len(dataset) - pivot])
			x1 = abs(max(group1) - data)
			x2 = abs(min(group2) - data)
			if x1 < x2:
				group1.append(data)
			else:
				group2.append(data)

	if depth == maxDepth:
		return [group1, group2]

	return splitGroups(group1, depth + 1, maxDepth) + splitGroups(group2, depth + 1, maxDepth)


def categorizeGroups(dataset, depth):
	## Iterate to extract every city
	l = len(dataset.values)//2
	delta = 5000
	# times = set(dataset.values[l:l+delta])
	times = set(dataset.values)

	## convert the set in an array
	orderedTimes = list(times)
	orderedTimes = sorted(orderedTimes)

	groups = splitGroups(orderedTimes, 1, depth)
	return groups


## load the iris data into a DataFrame from web
url = 'travel_dataset.csv' 

## Specifying column names.
col_names = ['state_p', 'city_p', 'state_c', 'city_c', 'normalized_next_time', 'next_time', 'route_part', 'total_time', 'theft_prob', 'theft_status']
routes = pd.read_csv(url, header=None, names=col_names)

colors = ['b', 'g', 'r', 'c', 'm', 'y', 'lightcoral', 'black', 'darkorange', 'purple']
r = routes['theft_prob']
groups = categorizeGroups(r, 1)

# dataset view 
# for i in range(0, len(groups)):
# 	lst = np.array(groups[i])
# 	print(lst.shape)
# 	it = range(len(groups[i]))
# 	g = groups[i]
# 	plt.scatter(it, g, color=colors[i])

# plt.show()

proc = routes[(routes['theft_prob'].isin(groups[1]))]
# proc = proc['total_time']
# it = range(len(proc))
# g = proc
# print(len(it), len(g))
# plt.scatter(it, g, color=[.5,.5,.5])
# plt.show()






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
from keras.layers import Dense, LSTM, Dropout, TimeDistributed
from keras.callbacks import EarlyStopping, TensorBoard
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
dataset = proc['theft_status']
dataset = [[a] for a in dataset]

y_dataset = proc['state_p']
y_dataset = [[a] for a in y_dataset]

# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

scaler_y = MinMaxScaler(feature_range=(0,1))
y_dataset = scaler_y.fit_transform(y_dataset)


# dataset = numpy.array([[(2*x + 1)/(2*99+1)] for x in range(0, 100)])
# dataset = numpy.array([[(0.3)*(-1)**x] for x in range(0, 200)])

# split into train and test sets
train_size = int(len(dataset) * 0.6)
test_size = int(len(dataset)*0.1)
valid_size = int(len(dataset)*0.3)
train, test, valid = dataset[0:train_size,:], dataset[train_size:train_size + test_size,:], dataset[train_size + test_size:,:]
y_train, y_test, y_valid = y_dataset[0:train_size,:], y_dataset[train_size:train_size + test_size,:], y_dataset[train_size + test_size:,:]

# reshape into X=t and Y=t+1
look_back = 5
# trainX, trainY = create_dataset(train, look_back)
# testX, testY = create_dataset(test, look_back)
# validX, validY = create_dataset(valid, look_back)

trainY, _ = create_dataset(train, 1)
testY, _ = create_dataset(test, 1)
validY, _ = create_dataset(valid, 1)

trainX, _ = create_dataset(y_train, look_back)
testX, _ = create_dataset(y_test, look_back)
validX, _ = create_dataset(y_valid, look_back)

# reshape input to be [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
validX = numpy.reshape(validX, (validX.shape[0], 1, validX.shape[1]))

# trainX = numpy.reshape(trainX, (trainX.shape[0], 5, 1))
# testX = numpy.reshape(testX, (testX.shape[0], 5, 1))
# validX = numpy.reshape(validX, (validX.shape[0], 5, 1))
trainY = trainY[abs(len(trainX)-len(trainY)):]
testY = testY[abs(len(testX)-len(testY)):]
validY = validY[abs(len(validX)-len(validY)):]


#############################################
## Define the model
#############################################
# create and fit the LSTM network
model = Sequential()
# model.add(LSTM(4, input_shape=(1, look_back), return_sequences=True))

model.add(LSTM(4, input_shape=(1, look_back), return_sequences=True))
model.add(LSTM(16, return_sequences=True))
model.add(LSTM(1, return_sequences=False))
model.add(Dropout(rate=0.3))
model.add(Dense(3))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

model.summary()
print("--------------------------------------")
print(trainX)
print("--------------------------------------")

# Early stopping callback
PATIENCE = 40
early_stopping = EarlyStopping(monitor='loss', min_delta=0, patience=PATIENCE, verbose=0, mode='auto')
callbacks = [early_stopping]

print(trainX.shape, trainY.shape)
history = model.fit(trainX, trainY, epochs=1000, batch_size=5, validation_data=(validX, validY), callbacks=callbacks, verbose=2)

# list all data in history
print(history.history.keys())

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
validPredict = model.predict(validX)


## Normalize the predictions
# trainPredict = trainPredict/max(trainPredict)
# testPredict = testPredict/max(testPredict)

# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])
validPredict = scaler.inverse_transform(validPredict)
validY = scaler.inverse_transform([validY])

# calculate root mean squared error
# trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
# print('Train Score: %.2f RMSE' % (trainScore))
# testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
# print('Test Score: %.2f RMSE' % (testScore))

model_json = model.to_json()
with open("models/model-theft-series.json", "w") as json_file:
    json_file.write(model_json)

model.save_weights("models/model-theft-series.h5")
print("Saved model to disk")



# shift train predictions for plotting
trainPredictPlot = numpy.empty_like(dataset_y)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

# shift test predictions for plotting
testPredictPlot = numpy.empty_like(dataset_y)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(trainPredict)+len(testPredict), :] = testPredict

# shift valid predictions for plotting
validPredictPlot = numpy.empty_like(dataset_y)
validPredictPlot[:, :] = numpy.nan
validPredictPlot[len(trainPredict)+len(testPredict):, :] = validPredict

# plot baseline and predictions
plt.plot(scaler.inverse_transform(dataset_y))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.plot(validPredictPlot)
plt.show()