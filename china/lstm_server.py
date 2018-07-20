#!/usr/bin/env python
# -*- coding: utf-8 -*-
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

print("Preparing dataset...")

## load the iris data into a DataFrame from web
url = 'travel_dataset.csv' 

## Specifying column names.
col_names = ['state_p', 'city_p', 'state_c', 'city_c', 'normalized_next_time', 'next_time', 'route_part', 'total_time', 'theft_prob', 'theft_status']
routes = pd.read_csv(url, header=None, names=col_names)

colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'black']
r = routes['total_time']
# groups = categorizeGroups(r, 3)
groupNum = 1
groups = categorizeGroups(r, 3)

proc = routes[(routes['total_time'].isin(groups[groupNum]))]

# import random

# for i in range(0, len(groups)):
# 	g = random.sample(groups[i], len(groups[i]))
# 	it = range(len(g))
# 	plt.scatter(it, g, color=colors[i])

# plt.show()
# exit()

# allGroups = [] ## Variavel global para ser usada no servidor
# for i in range(1, len()):
# 	allGroups.append(routes[(routes['total_time'].isin(groups[i]))])

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
from keras import backend as K
from keras.models import model_from_json

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
cacheDataset = dataset.copy()

# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
# dataset = scaler.fit_transform(dataset)

# split into train and test sets
train_size = int(len(dataset) * 0.6)
test_size = int(len(dataset)*0.1)
valid_size = int(len(dataset)*0.3)

# reshape into X=t and Y=t+1
time_steps = 1
look_back = 5
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

print("Preparing network...")

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

def completeRoute(begin, end, groupNumCache, pathCache):
	## pega o grupo
	name = str(begin) + "/" + str(end)
	groupNum = groupNumCache[name]
	path = pathCache[groupNum][name]
	## pega a rota naquele grupo
	return groupNum, path

def cacheAllPairs():
	## Specifying column names.
	col_names = ['state_p', 'city_p', 'state_c', 'city_c', 'normalized_next_time', 'next_time', 'route_part', 'total_time', 'theft_prob', 'theft_status', 'inicio/fim', 'caminhotodo']
	url = 'travel_dataset_extra.csv' 
	routes = pd.read_csv(url, header=None, names=col_names)

	## Coisas importantes de retornar
	routeToGroup = {}
	completeRouteByGroup = [dict() for _ in range(0,9)]
	maxTimeByGroup = [0 for _ in range(0,9)]

	for groupNum in range(1,len(groups)):
	# for groupNum in range(1,2):
		print("Extracting routes for group", groupNum)
		myset = routes[(routes['total_time'].isin(groups[groupNum]))]
		# proc = routes[(routes['total_time'].isin(groups[groupNum]))] ## Pega o grupo da variavel global que contém os grupos

		#########################################################
		## Here we extract the usefull data for us
		#########################################################
		rotasDoEstado = [(iniciofim, frame) for iniciofim, frame in myset.groupby('inicio/fim')] # separa em conjuntos pelo tempo total

		print("computing time...")
		## Computes the max amount of time
		maxTime = 0
		for df in frameSet:
			total_time, frame = df
			if total_time > maxTime:
				maxTime = total_time

		maxTimeByGroup[groupNum] = maxTime

		print("caching routes...")
		## Join the time and normalize it
		for dataFrame in rotasDoEstado:
			iniciofim, frame = dataFrame
			todoCaminho = numpy.array(frame["caminhotodo"])

			caminho = todoCaminho[len(todoCaminho)//2] ## Escolhe um caminho qualquer do conjunto
			routeToGroup[iniciofim] = groupNum
			completeRouteByGroup[groupNum][iniciofim] = caminho

	return routeToGroup, completeRouteByGroup


print("Setting up city conversor...")
translateState = getStateIdentifiers("../paths/path/")

## Generate all the route pairs
print("Caching all possible routes...")
routeToGroupCache, completeRouteByGroupCache = cacheAllPairs()


# val = completeRoute(begin, end, routeToGroupCache, completeRouteByGroupCache)
	


#########################################################
## Flask part of the script
#########################################################
print("Launching server...")
import random
import json

import os
from os import listdir
from os.path import isfile, join
import codecs
from time import sleep
import datetime

## Handle the network
import io
import json
import requests
from datetime import datetime
from flask import Flask, request
from flask_cors import CORS

## Handle with the operating system
import sys
from pathlib import Path

## Handle charset
from pprint import pprint
from unicodedata import normalize

## Crypto
from itertools import islice, count
from Crypto.Util import number

app = Flask(__name__)
CORS(app) # Allow all domains

privateNum = {}
publicPrime = {}
generator = {}
sharedKey = {}

## Todos os estados do brasil
states = {
	'Rio de Janeiro': 10600, 
	'São Paulo': 10584,
	'Rio Grande do Sul': 1144,
	'Minas Gerais': 588,
	'Goiás': 529,
	'Bahia': 323,
	'Ceará': 229,
	'Santa Catarina': 206,
	'Pernambuco': 201,
	'Pará': 117,
	'Mato Grosso': 98,
	'Distrito Federal': 82,
	'Rio Grande do Norte': 79,
	'Alagoas': 77,
	'Espírito Santo': 71,
	'Maranhão': 43,
	'Tocantins': 18,
	'Piauí': 6,
	'Paríba': 5,
	'Rondônia': 4,
	'Mato Grosso do Sul': 4,
	"Amazonas": 2,
	'Roraima': 2,
	'Sergipe': 1,
	'Acre': 0,
	'Amapá': 0,
	'Paraná': 0
}

## Lista das capitais do brasil
capitais_brasil = [
	"Belem,%20PA",
	"Porto%20Velho,%20RO",
	"Boa%20Vista,%20RR",
	"Manaus,%20AM",
	"Palmas,%20TO",
	"Maceio,%20AL",
	"Salvador,%20BA",
	"Fortaleza,%20CE",
	"Sao%20Luis,%20MA",
	"Joao%20Pessoa,%20PB",
	"Recife,%20PE",
	"Teresina,%20PI",
	"Natal,%20RN",
	"Aracaju,%20SE",
	"Goiânia,%20GO",
	"Cuiaba,%20MT",
	"Campo%20Grande,%20MS",
	"Brasilia,%20BR",
	"Vitoria,%20ES",
	"Belo%20Horizonte,%20MG",
	"Sao%20Paulo,%20SP",
	"Rio%20de%20Janeiro,%20RJ",
	"Curitiba,%20PR",
	"Porto%20Alegre,%20RS",
	"Florianopolis,%20SC"
]



################################################################################
## Converte a lista de relação de roubo em uma probabilidade de algo ser
## roubado. O número é dado por:
## summer calcula o inverso do número de trajetos, ou seja, o número de
## trajetos no país nessas estatísticas foi: sum(states.values())*88
## A probabilidade de ser roubado por estado foi estimada dividindo o número  
## de roubos que houve no estado pelo número de trajetos realizados no país
## o que tornaria muito mais coerente a questão da probabilidade de roubos em
## um determinado local, supondo que houveram entregas igualmente distribuidas
## ao longo de toda nação que não é algo extremamente preciso mas serve para  
## os fins discutidos aqui nesse projeto
################################################################################
summer = 1.0/sum(states.values())/88
norm = {k: v*summer for k, v in states.items()}

## TODO: Construir uma lista com os valores de transporte para cada país
## fazer em seguida listeners para mudar esses valores
amount = sum(states.values())*88//len(states)
totalTrips = dict()

for key, value in states.items():
	totalTrips[key] = (value, amount)

print(totalTrips)

################################################################################
## Auxiliary functions
################################################################################
def getProb(key):
	"""
	Recupera a probabilidade de um estado ser roubado
	"""
	prob = float(totalTrips[key][0])/float(totalTrips[key][1])
	return prob

def modifyProb(key, value):
	"""
	Insere elementos no array e roubos
	"""
	totalTrips[key][0] += value
	totalTrips[key][1] += value

def getRouteProb(keys):
	"""
	Calcula a probabilidade de roubo em uma determinada rota
	"""
	prod = 1.0
	for key in keys:
		prod *= 1-getProb(key)
	return 1-prod

def getRouteTime(route, groupNum=1):
	time = -1
	rt = np.array(route)
	K.clear_session()
	rt = numpy.reshape(rt, (1, 1, 5))

	modelName = "models/model-temporal-series-zone-"+str(groupNum)
	json_file = open(modelName+".json", 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	model = model_from_json(loaded_model_json)

	model.load_weights(modelName+".h5")
	time = model.predict(rt)
	return time[0][0]

def routeFromString(routeString):
	states = routeString.split("/")
	route = [translateState[a]for a in states[:-1]]
	while len(route) > 5:
		route = route[:-1]

	while len(route) < 5:
		route += [route[-1]]

	return route

################################################################################
## Record the position                                                        ##
################################################################################
@app.route('/', methods=['GET'])
def listenner():
	return "server working", 200

@app.route('/getTime', methods=['GET', 'POST'])
def getTime():
	if request.method == 'GET':
		route = [2, 3, 4, 5, 6]
		## TODO: redefinir o método para obter o número da rede a ser usada
		groupNum = 1
		time = getRouteTime(route, groupNum)
		return "server funcionando!!!"+" -> "+str(time), 200

	print("received request!")
	args = request.get_json()
	rota = args["rota"]
	rota = rota.split("/")

	begin = rota[0]
	end = rota[1]
	num, routeStr = completeRoute(begin, end, routeToGroupCache, completeRouteByGroupCache)
	route = routeFromString(routeStr)

	groupNum = 1
	time = getRouteTime(route, num)*maxTime
	res = str(time)
	return res, 200


if __name__ == '__main__':
    # myport = int(os.getenv('PORT', 8081))
    myport = 8081
    print("Starting app on port %d" % myport)
    app.run(host='0.0.0.0', port=myport, debug=True)


