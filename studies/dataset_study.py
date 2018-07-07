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

colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'black']
r = routes['total_time']
groups = categorizeGroups(r, 3)

# dataset view 
plt.scatter(routes['total_time'][::40], routes['next_time'][::40])
for i in range(0, len(groups)):
	it = range(len(groups[i]))
	g = groups[i]
	plt.scatter(it, g, color=colors[i])

plt.show()

# proc = routes[(routes['total_time'].isin(groups[3]))]
# proc = proc['total_time']
# it = range(len(proc))
# g = proc
# print(len(it), len(g))
# plt.scatter(it, g, color=[.5,.5,.5])
# plt.show()


