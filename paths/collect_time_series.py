#########################################################
## this software is intend to generate a time series from
## the data list present in the text files
#########################################################
import os

basedir = "path/"
lst = os.listdir(basedir)
base_num = 27869
dataset = []

def get_data_in_folder(basedir):
	"""
	Extract all data from the files in the MarkMe AI project in a folder
	"""
	if basedir[-1] != "/":
		basedir += "/"

	lst = os.listdir(basedir)
	data = []

	## this script extract all the data into an array to group the data to train
	for name in lst: # iterate in every entry of the folder
		with open(basedir + name) as f: # open the file in the dir
			fileData = f.readlines() # get all lines from a file as an array, the 
			path = [point.split(" - ") for point in fileData]
			data += [path]

	return data

def get_cities_list(named="cities"):
	"""
	Get all the cities in the set, and enumerate them to be used
	as input data in the system

	@parameter named: what is intended to add in the list, cities
	or states as a string
	"""
	selector = 0
	if named == "cities":
		selector = 0
	elif named == "states":
		selector = 1


	paths = get_data_in_folder("path/")


	## Iterate to extract every city
	cities = set()

	for path in paths:
		for line in path:
			if len(line) > 1:
				cities.add(line[selector])
				# if line[2] == 1: ## locate the occurrency of a theft
				# 	print("theft at:", line)

	## convert the set in an array
	arr = list(cities)
	arr = sorted(arr)
	return arr

def make_inverted_dict(lst):
	res = dict()

	ct = 0
	for a in lst:
		res[a] = ct
		ct += 1
	return res

def get_max_time():
	"""
	Get the highest time in course from the entire dataset
	"""
	paths = get_data_in_folder("path/")
	highest = 0

	for path in paths:
		highest = max(highest, float(path[-1][0])) ## for some reason path[-1] is a list

	return highest

#########################################################
## For now get all the times in the dataset of time
## series
#########################################################
city = get_cities_list("cities")
city = make_inverted_dict(city)

state = get_cities_list("states")
state = make_inverted_dict(state)

max_time = get_max_time()

## this script extract all the data into an array to group the data to train
for name in lst: # iterate in every entry of the folder
	
	## Ignore the data that does not have the partial times
	num = int(name.split(".")[0])
	if num < base_num:
		continue

	with open(basedir + name) as f: # open the file in the dir
		data = f.readlines() # get all lines from a file as an array, the 
		
		# in data:
		# all the first lines has a shape of the spot
		# the last one has the total time spent in traveling throught this route, is problematic
		# because like that there is no way for us to make the path time real prediction
		# more data must be collected here
		# print(data)

		# list with the shape:
		# 0 - city
		# 1.- state
		# 2 - probability of be stollen
		# 3 - is stolen or not
		# 4 - the partial time of the delivery (just is some of the data)
		path = [point.split(" - ") for point in data]

		ct = 0 # count to know when is the first element of the iteration

		for p in path[:len(path) - 2]:
			if len(p) == 5:
				#########################################################
				## The dataset will work as follows:
				## 0 - the state where the guy came from
				## 1 - the city where the guy came from
				## 2 - the state where the guy is going
				## 3 - the city where the guy is going at all
				## 4 - the amount of time necessary to the guy get in 
				## there, normalized by the highest amount in the dataset
				## 5 - the real amount of time necessary
				## 6 - the partial time in the route
				## 7 - the time of the entire set
				## 
				## Also in this dataset the data will be grouped in
				## groups of five edges, those edges are the entire
				## route at all
				## OBS: in order to keep the dataset coherent the time
				## spent in the first city will be considered as stock
				## time, it is, the time spent untill the delivery was
				## packaged and sent to the user itself. (-1 -> storage)
				#########################################################
				if ct == 0: ## Threat the case of the storage problem in the dataset
					city_p = -1
					state_p = -1
				else:
					city_p = city[path[ct - 1][0]]
					state_p = state[path[ct - 1][1]]
				city_c = city[p[0]]
				state_c = state[p[1]]

				num = float(p[4].split("\n")[0])
				tm = float(path[-1][0])
				time_p = num*tm
				time_t = time_p/max_time


				dataset += [[state_p, city_p, state_c, city_c, time_t, time_p, num, tm]]
				ct += 1


print(dataset[:100])

with open("travel_dataset.csv", "w") as f:
	for data in dataset:
		string = ",".join(str(x) for x in data)
		f.write(string + "\n")
		# f.write(str(data[0]) + "," + str(data[1]) + "\n")
	