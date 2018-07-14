
#############################################
## Import all libs
#############################################
# LSTM for international airline passengers problem with regression framing
import numpy
import matplotlib.pyplot as plt
from pandas import read_csv
import math
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


########################################################
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

colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'black']
r = routes['total_time']
groups = categorizeGroups(r, 3)

# # dataset view 
# plt.scatter(routes['total_time'][::40], routes['next_time'][::40])
# for i in range(0, len(groups)):
# 	it = range(len(groups[i]))
# 	g = groups[i]
# 	plt.scatter(it, g, color=colors[i])

# plt.show()

proc = routes[(routes['total_time'].isin(groups[3]))]


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
u = [[a] for a in dataset]
dataset = [[a] for a in dataset]

# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

# split into train and test sets
test_size = len(dataset) - 1
test = dataset[0:test_size,:]

# reshape into X=t and Y=t+1
look_back = 1
testX, testY = create_dataset(test, look_back)

# reshape input to be [samples, time steps, features]
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
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
# model = load_model('model-temporal-series1.h5')
model.load_weights('model-temporal-series1.h5')

# make predictions
testPredict = model.predict(testX)

# invert predictions
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

# calculate root mean squared error
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))

# shift test predictions for plotting
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[(look_back*2)+1:test_size + 1, :] = testPredict

# plot baseline and predictions
plt.plot(scaler.inverse_transform(dataset[:test_size, :]))
plt.plot(testPredictPlot)
plt.show()

