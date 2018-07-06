
#########################################################
# This script runs in 1 second without prints
#########################################################

import os


basedir = "path/"
lst = os.listdir(basedir)
base_num = 27869

## this script extract all the data into an array to group the data to train
for name in lst: # iterate in every entry of the folder
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
		for p in path[:len(path) - 2]:
			if len(p) == 5:
				if p[4] != 0:
					print(name)
					exit()
			# if p[3] == "1":
			# 	print("stolen at: ", name, p)
		# print("-------------------------------------------")
		# print(path)

		# break
