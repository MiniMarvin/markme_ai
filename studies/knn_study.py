import matplotlib.pyplot as plt
import pandas as pd

## load the iris data into a DataFrame from web
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data' 

## Specifying column names.
col_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
iris = pd.read_csv(url, header=None, names=col_names)

## dataset view
def datasetView(iris):
	_, g1 = [x for _, x in iris.groupby(iris['species'] == 'Iris-setosa')]
	_, g2 = [x for _, x in iris.groupby(iris['species'] == 'Iris-versicolor')]
	_, g3 = [x for _, x in iris.groupby(iris['species'] == 'Iris-virginica')]
	plt.scatter(g1['petal_width'], g1['petal_length'], color="red")
	plt.scatter(g2['petal_width'], g2['petal_length'], color="green")
	plt.scatter(g3['petal_width'], g3['petal_length'], color="blue")
	# plt.scatter(iris['petal_width'], iris['petal_length'], color="blue")

	plt.show()

## map each iris species to a number with a dictionary and list comprehension.
iris_class = {'Iris-setosa':0, 'Iris-versicolor':1, 'Iris-virginica':2}
iris['species_num'] = [iris_class[i] for i in iris.species]


## Create an 'X' matrix by dropping the irrelevant columns.
X = iris.drop(['species', 'species_num'], axis=1)
y = iris.species_num

from sklearn.model_selection import train_test_split
## Split data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

## Import the Classifier.
from sklearn.neighbors import KNeighborsClassifier
## Instantiate the model with 5 neighbors. 
knn = KNeighborsClassifier(n_neighbors=5)
## Fit the model on the training data.
knn.fit(X_train, y_train)
## See how the model performs on the test data.
score = knn.score(X_test, y_test)

datasetView(iris)
