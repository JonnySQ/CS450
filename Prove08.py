import os
import sys
import math
import numpy as np
import pandas as pd

from sklearn import datasets
from sklearn import utils
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

# CLASSES

# Classifier itself
class HardCodedClassifier():
    def __init__(self):
       pass
    
    def fit(self, trainData, trainTarget):
       #How many layers?
       numLayers = int(input("How many layers: "))
       numNodes = []

       for i in range(numLayers):
          ask = "Number of nodes in layer " + str((i + 1)) + ": "
          indNum = int(input(ask))
          numNodes.append(indNum)

       #create random data
       attributes = int(len(trainData[0]) + 1)

       #create random weights
       weights = []   #create random weights using given number of layers and nodes in each layer
                      #total layers /  #number weights

       for i in range(numLayers):
          nodeWeights = []
          for j in range(numNodes[i] - 1):
             nodeWeights.append(np.random.randn(attributes))
          weights.append(nodeWeights)

       inputsExtend = [-1] #begins with the bias value, used to extend layered inputs

       for h in range(len(trainData)):
          layeredInputs = []                   #[[input layer], [layer 1], ... [output layer]]
          for i in range(len(trainData[h])):
             inputsExtend.append(trainData[h][i])
          layeredInputs.append(inputsExtend)
          inputsExtend = [-1]                  # reset for other layers
   
          for i in range(numLayers):
             for j in range(numNodes[i] - 1):
                inputsExtend.append(Layer(layeredInputs[i], weights[i][j]).findNode())
             layeredInputs.append(inputsExtend)
             inputsExtend = [-1]                #reset with only bias value
          print(layeredInputs)
       
       model = Model()
       return model
    
# Mode to user for the classifier
class Model():
    def __init__(self):
       pass
       
    def predict(self, testData):
       return 0
       
# Representing each layer including input, output, and hidden
class Layer():
    def __init__(self, values, weights):
       self.nodes = []
       for i in range(len(values)):
          self.nodes.append(Node(values[i], weights[i]))
    
    def findNode(self):
       total = 0
       for i in range(len(self.nodes)):
          total = total + (self.nodes[i].value * self.nodes[i].weight)
       output = sigmoid(total)
       return output

# Representing each individual node
class Node():
    value = 0  #value of the node itself
    weight = 0 #weight value
    
    def __init__(self, value, weight):
       self.value = value
       self.weight = weight

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def prepIris(irisData):
   minMax = preprocessing.MinMaxScaler()
   normIrisData = minMax.fit_transform(irisData)
   return normIrisData

#Prepare data for the indian data set
def indianDataPrep(dataSet):
    for i in range(len(dataSet[0])):
        maxValue = dataSet[0][i]
        minValue = dataSet[0][i]
        for j in range(len(dataSet)):
            if (dataSet[j][i] > maxValue):
                maxValue = dataSet[j][i]
            if (dataSet[j][i] < minValue):
                minValue = dataSet[j][i]
        
        totalValue = maxValue - minValue
        
        for j in range(len(dataSet)):
            dataSet[j][i] = float(float(dataSet[j][i] - minValue) / totalValue)
        
    indianTarget = [i[8] for i in dataSet]
    indianData = [i[0:7] for i in dataSet]
    
    print(indianData)
    print(indianTarget)
    
    return indianTarget, indianData

#MAIN FUNCTION

# Load and prep the data
iris = datasets.load_iris()
irisData = prepIris(iris.data)

df = pd.read_table('pima-indians-diabetes.data.txt', sep=',')
fullData = df.values
indianTarget, indianData = indianDataPrep(fullData)

# Randomize split into a training set: (70%) and testing set (30%)
iris_train1, iris_test1, iris_train2, iris_test2 = train_test_split(irisData, iris.target, test_size=0.30)

indian_train1, indian_test1, indian_train2, indian_test2 = train_test_split(indianData, indianTarget, test_size=0.30)

# Create model (Hardcoded)
classifier = HardCodedClassifier()
model = classifier.fit(iris_train1, iris_test1)

classifierIndian = HardCodedClassifier()
modelIndian = classifierIndian.fit(indian_train1, indian_train2)

# Hardcoded Predictions
match = 0
for i in range(len(iris_test2)):
   one = iris_test2[i]
   two = model.predict(iris_test2[i])
   if one == two:
       match += 1

matchIndian = 0
for i in range(len(indian_test2)):
   one = indian_test2[i]
   two = modelIndian.predict(indian_test2[i])
   if one == two:
       matchIndian += 1

#Present accuracy percentages
print("Iris: "  + str(match) + "/" + str(len(iris_test2)))
print("Indian: " + str(matchIndian) + "/" + str(len(indian_test2)))
