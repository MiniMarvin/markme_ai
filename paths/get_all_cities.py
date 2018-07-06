import os


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


paths = get_data_in_folder("path/")

## Iterate to extract every city
cities = set()

for path in paths:
	for line in path:
		if len(line) > 1:
			cities.add(line[0])
			cities.add(line[1])
			if line[2] == 1: ## locate the occurrency of a theft
				print("theft at:", line)



## convert the set in an array
arr = list(cities)
arr = sorted(arr)

print(len(arr))
print(arr[0:10])
