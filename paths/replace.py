import os
from os import listdir
from os.path import isfile, join



for i in range(1, 100): # iterate in every folder
	mypath = "path" + str(i) + "/"

	if os.path.exists(mypath): # if the path exists move the path to the base dir
		lst = os.listdir("path/") # dir is your directory path
		n = len(lst)
		nxt = os.listdir(mypath)
		# for j in range(0, 500): # iterate in every file in the folder and moves it to the base folder
			# os.system("mv " + mypath + str(j) + ".txt" + " " + "path/" + str(n + j) + ".txt"  )
		j = 0
		for file in nxt: # iterate in every file in the folder and moves it to the base folder
			os.system("mv " + mypath + file + " " + "path/" + str(n + j) + ".txt")
			j += 1