#########################################################
## This piece of software groups data into batches based
## in the time they take to process
##
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

proc = routes[(routes['total_time'].isin(groups[3]))]


#########################################################
## Here we extract the usefull data for us
#########################################################
frameSet = [(total_time, frame) for total_time, frame in proc.groupby('total_time')]

groupset = []
for dataFrame in frameSet:
	total_time, frame = dataFrame
	lst = frame['state_p'].values
	
	while len(lst) > 5:
		lst.pop()

	while len(lst) < 5:
		lst = np.append(lst, -1)

	groupset += [(lst, total_time)]


#########################################################
## Here we build a CNN to process this data
#########################################################
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense, Conv1D, MaxPooling1D
from keras.callbacks import EarlyStopping, TensorBoard
from sklearn.metrics import accuracy_score, f1_score
from datetime import datetime

inputShape = groupset[0][0].shape

model = Sequential()
model.add(Dense(10, activation=None, input_shape=inputShape))
model.add(Dense(100, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(200, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(200, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(200, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(200, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(200, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(200, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(100, activation="relu"))
model.add(Dense(1))
# model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
model.compile(loss='mean_squared_error',optimizer='rmsprop',metrics=['accuracy'])
model.summary()


train = groupset[:int(0.6*len(groupset))] # 60% to train
validate = groupset[int(0.6*len(groupset)):int(0.9*len(groupset))] # 30% to test
test = groupset[int(0.9*len(groupset)):] # 10% to validate

trainX, trainY = zip(*train)
testX, testY = zip(*test)
validateX, validateY = zip(*validate)

trainX, trainY = np.array(trainX), np.array(trainY)
testX, testY = np.array(testX), np.array(testY)
validateX, validateY = np.array(validateX), np.array(validateY)

epochs = 1000
batch_size = 400

# Early stopping callback
PATIENCE = 40
early_stopping = EarlyStopping(monitor='loss', min_delta=0, patience=PATIENCE, verbose=0, mode='auto')

# TensorBoard callback
LOG_DIRECTORY_ROOT = 'logs/'
now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
log_dir = "run-{}/".format(LOG_DIRECTORY_ROOT, now)
tensorboard = TensorBoard(log_dir=log_dir, write_graph=True, write_images=True)

# Place the callbacks in a list
# callbacks = [early_stopping, tensorboard]
callbacks = [tensorboard]

# Train the model
history = model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, validation_data=(validateX, validateY), callbacks=callbacks, verbose=5)

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

test_predictions = model.predict(testX)

plt.plot(testY)
plt.plot(test_predictions)
plt.show()

# Report the accuracy
accuracy = model.evaluate(testX, testY, batch_size=200)
print("Accuracy: " + str(accuracy))




