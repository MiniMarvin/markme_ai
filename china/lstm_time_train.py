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
	group1 = [max(dataset)] # max
	group2 = [min(dataset)] # min
	min_at_all = group1[0]
	max_at_all = group1[0]

	selector = [1, -1]

	## Goes computing the difference between the two groups and join to the list 
	## where the valueapproaches best

	for pivot in range(0, len(dataset)):
		data = dataset[pivot]
		x1 = abs(data - min_at_all)
		x2 = abs(data - max_at_all)
		if data > max_at_all:
			max_at_all = data
			group1.append(data)
		elif data < min_at_all:
			min_at_all = data
			group2.append(data)

		elif x1 > x2:
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

colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'black']
r = routes['total_time']
# groups = categorizeGroups(r, 3)
groupNum = 2
groups = categorizeGroups(r, 3)

proc = routes[(routes['total_time'].isin(groups[groupNum]))]

#############################################
## Import all libs
#############################################
# LSTM for international airline passengers problem with regression framing
import numpy
import matplotlib.pyplot as plt
from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
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
dataset = proc['normalized_next_time']
dataset = [[a] for a in dataset]

xdataset = proc['state_c']
xdataset = [[a] for a in xdataset]

#########################################################
## Here we extract the usefull data for us
#########################################################
frameSet = [(total_time, frame) for total_time, frame in proc.groupby('total_time')] # separa em conjuntos pelo tempo total

## Computes the max amount of time
maxTime = 0
for df in frameSet:
	total_time, frame = df
	if total_time > maxTime:
		maxTime = total_time

## Join the time and normalize it
groupset = []
for dataFrame in frameSet:
	total_time, frame = dataFrame
	total_time = total_time/maxTime
	lst = frame['state_c'].values

	# print(total_time)

	while len(lst) > 5:
		lst = lst[:-1]

	while len(lst) < 5:
		lst = np.append(lst, -1)

	groupset += [(np.array(lst), total_time)]

dataset = groupset

# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
# dataset = scaler.fit_transform(dataset)

# split into train and test sets
train_size = int(len(dataset) * 0.6)
test_size = int(len(dataset)*0.1)
valid_size = int(len(dataset)*0.3)
# train, test, valid = dataset[0:train_size,:], dataset[train_size:train_size + test_size,:], dataset[train_size + test_size:,:]

# reshape into X=t and Y=t+1
time_steps = 1
look_back = 5
np.random.shuffle(dataset)

datasetX, datesetY = zip(*dataset)
trainX, trainY = zip(*dataset[:train_size])
trainX, trainY = np.array(trainX), np.array(trainY)
testX, testY = zip(*dataset[train_size:train_size+test_size])
testX, testY = np.array(testX), np.array(testY)
validX, validY = zip(*dataset[train_size+test_size:train_size+test_size+valid_size])
validX, validY = np.array(validX), np.array(validY)

# reshape into X=t and Y=t+1
trainX = numpy.reshape(trainX, (trainX.shape[0]//time_steps, time_steps, trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0]//time_steps, time_steps, testX.shape[1]))
validX = numpy.reshape(validX, (validX.shape[0]//time_steps, time_steps, validX.shape[1]))

print("---------------------------------")
print(testX, testY)
print("---------------------------------")

#############################################
## Define the model
#############################################
# create and fit the LSTM network
model = Sequential()
model.add(LSTM(40, input_shape=(1, look_back), return_sequences=False))
model.add(Dropout(rate=0.3))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='rmsprop')

# Early stopping callback
PATIENCE = 100
early_stopping = EarlyStopping(monitor='loss', min_delta=0, patience=PATIENCE, verbose=0, mode='auto')
callbacks = [early_stopping]

modelName = "models/model-temporal-series-zone-"+str(groupNum)
#####################################
## UNCOMMENT
#####################################
history = model.fit(trainX, trainY, epochs=100, batch_size=5, validation_data=(validX, validY), callbacks=callbacks, verbose=2)

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

model_json = model.to_json()
with open(modelName + ".json", "w") as json_file:
    json_file.write(model_json)

model.save_weights(modelName+".h5")
print("Saved model to disk")
#####################################
model.load_weights(modelName+".h5")

# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
validPredict = model.predict(validX)

print("--------------------------------------")
trainY = np.array([trainY])
testY = np.array([testY])
print("predict: ", trainPredict[0])
print("labels: ", trainY[0])
print("--------------------------------------")

#####################################
## UNCOMMENT
#####################################
# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))
#####################################

# shift train predictions for plotting
# trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot = numpy.zeros((len(dataset), 1))
print(trainPredictPlot.shape)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[:train_size, :] = trainPredict

# shift test predictions for plotting
testPredictPlot = numpy.zeros((len(dataset), 1))
testPredictPlot[:, :] = numpy.nan
testPredictPlot[train_size:train_size+test_size, :] = testPredict

# shift test predictions for plotting
validPredictPlot = numpy.zeros((len(dataset), 1))
validPredictPlot[:, :] = numpy.nan
validPredictPlot[train_size+test_size:train_size+test_size+valid_size, :] = validPredict

#####################################
## UNCOMMENT
#####################################
# plot baseline and predictions
plt.plot(datesetY)
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.plot(validPredictPlot)
plt.show()
#####################################






