#########################################################
## this software is intend to generate a time series from
## the data list present in the text files
#########################################################
import os

def get_data_in_folder(basedir="../paths/path/"):
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

def get_cities_list(named="cities", basedir="../paths/path/"):
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


	paths = get_data_in_folder(basedir)


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

#########################################################
## For now get all the times in the dataset of time
## series
#########################################################
def getStateIdentifiers(basedir="../paths/path/"):
	lst = os.listdir(basedir)
	base_num = 27869
	dataset = []

	state = get_cities_list("states", basedir)
	state = make_inverted_dict(state)
	return state


translateState = getStateIdentifiers("../paths/path/")
