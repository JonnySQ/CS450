####################################################
# Prove 02, kNN Classifier
# Author: Jon Crawford
# Professor: Brother Burton
# Summary - using k-Nearest Neightbors algorithm
####################################################

import sys
import math
import numpy as np

from sklearn import datasets
from sklearn import utils
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier


#gets the distance between 2 data points for any number of dimensions
def distance(data1, data2):
    connectPoints = zip(data1, data2)
    squaredDistances = [pow(x - y, 2) for (x,y) in connectPoints]
    return math.sqrt(sum(squaredDistances))

# Classifier itself
class HardCodedClassifier():
    def __init__(self, k):
       self.k = k
    
    def fit(self, trainData, trainTarget):
       model = Model(trainData, trainTarget, k)
       return model
    
# Mode to user for the classifier
class Model():
    def __init__(self, trainData, trainTarget, k):
       self.trainData = trainData
       self.trainTarget = trainTarget
       self.k = k
    
    def predict(self, testData):
       #resulting array of predictions
       distances = np.zeros(len(self.trainData))
       prediction = np.zeros(len(testData))
       
       #Compare distances to all trainData to each trainTarget
       for i in range (len(testData)):
          for j in range (len(self.trainData)):
             #print(i, j, distance(testData[i], self.trainData[j]))
             distances[j] = distance(testData[i], self.trainData[j])
          
          zeros = 0
          ones = 0
          twos = 0
          
          for nbr in range(self.k):
             #print("*",distances.argmin(), self.trainTarget[distances.argmin()])
             #print(distances[distances.argmin()])
             #print("---")
             distances[distances.argmin()] = 100
             
             #tally up the votes
             if (self.trainTarget[distances.argmin()] == 0):
                 zeros += 1
             elif (self.trainTarget[distances.argmin()] == 1):
                 ones += 1
             elif (self.trainTarget[distances.argmin()] == 2):
                 twos += 1
          
          if zeros > ones and zeros > twos:
              prediction[i] = 0
          elif ones > zeros and ones > twos:
              prediction[i] = 1
          elif twos > ones and twos > zeros:
              prediction[i] = 2
              
       return prediction

# Load the data
iris = datasets.load_iris()

# Randomize split into a training set: (70%) and testing set (30%)
iris_train1, iris_test1, iris_train2, iris_test2 = train_test_split(iris.data, iris.target, test_size=0.30)

#GAUSSIANNB

# Create model (GaussianNB)
classifier1 = GaussianNB()
model1 = classifier1.fit(iris_train1, iris_train2)

# Make predictions
predictions1 = model1.predict(iris_test1)

# Gaussian Predictions
match1 = 0
for i in range(len(iris_test2)):
   one = iris_test2[i]
   two = predictions1[i]
   if one == two:
      match1 += 1

#For neighbors

k = int(input("input neighbors: "))

#SKLEARN

# Create model (skLearn)
classifier2 = KNeighborsClassifier(n_neighbors = k)
model2 = classifier2.fit(iris_train1, iris_train2)

# Make Predictions (skLearn)
predictions2 = model2.predict(iris_test1)

match2 = 0
for i in range(len(iris_test2)):
    one = iris_test2[i]
    two = predictions2[i]
    if one == two:
        match2 += 1

#HARDCODED
#iris_train1 - numbers (bigger)
#iris_test1 - actual results (bigger)
#iris_train2 - numbers (smaller)
#iris_test2 - actual results (smaller)

#print(len(iris_train1), "+", len(iris_train2), "+", len(iris_test1), "+", len(iris_test2))

# Create model (Hardcoded)
classifier3 = HardCodedClassifier(k)
model3 = classifier3.fit(iris_train1, iris_train2)

# Make Predictions (Hardcoded)
predictions3 = model3.predict(iris_test1)

match3 = 0
for i in range(len(iris_test2)):
   one = iris_test2[i]
   two = predictions3[i]
   if one == two:
       match3 += 1

#RESULTS

print("SkLearn: " + str(match1) + "/" + str(len(iris_test2)))
print("GaussianNB: " + str(match2) + "/" + str(len(iris_test2)))
print("Hardcoded: " + str(match3) + "/" + str(len(iris_test2)))

