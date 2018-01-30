####################################################
# Prove 03, kNN Classifier
# Author: Jon Crawford
# Professor: Brother Burton
# Summary - using k-Nearest Neightbors algorithm
####################################################

import os
import sys
import math
import numpy as np
import pandas as pd

#sklearn imports
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
          threes = 0
          fours = 0
          
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
             elif (self.trainTarget[distances.argmin()] == 3):
                 threes += 1
             elif (self.trainTarget[distances.argmin()] == 4):
                 fours += 1
          
          findMax = (zeros, ones, twos, threes, fours)
          actualMax = max(findMax)
          
          if actualMax == zeros:
              prediction[i] = 0
          elif actualMax == ones:
              prediction[i] = 1
          elif actualMax == twos:
              prediction[i] = 2
          elif actualMax == threes:
              prediction[i] = 3
          elif actualMax == fours:
              prediction[i] = 4
          else:
              prediction[i] = 4
          
       return prediction

# Prepare data for the car evaluation dataset
def carDataPrep(dataSet):
    
    for i in range(len(dataSet)):
       for j in range(len(dataSet[i])):
          if dataSet[i][j] in ('unacc', 'low', 'small', '1'):
              dataSet[i][j] = 1
          elif dataSet[i][j] in ('acc', 'med', '2'):
              dataSet[i][j] = 2
          elif dataSet[i][j] in ('good', 'high', 'big', '3'):
              dataSet[i][j] = 3
          elif dataSet[i][j] in ('vgood', 'vhigh', '4'):
              dataSet[i][j] = 4
          elif dataSet[i][j] in ('more', '5more', '5'):
              dataSet[i][j] = 5
              
    carTarget = [i[0] for i in dataSet]
    carData = [i[1:] for i in dataSet]

    return carTarget, carData
    
'''
    PREDICTING condition-> unacc 1, acc 2, good 3, vgood 4
    buying -> low 1, med 2, high 3, vhigh 4
    maint -> low 1, med 2, high 3, vhigh 4
    doors -> 2, 3, 4, 5more 5
    persons -> 2, 4, more 5
    lug_boot -> small 1, med 2, big 3
    safetey -> low 1, med 2, high 3
'''

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
        
    return indianTarget, indianData

'''
pregnant
plasma
diastolic
tricepts
2-hour
bmi
diabetes
age
PREDICTING class
'''

def mpgDataPrep(dataSet):
    return dataSet

'''
PREDICTING mpg
cylinders
displacement
horsepower
weight
acceleration
model year
origin
car name
'''

#ACTUAL PROGRAM STARTS

#WEEK 01 CONTENT

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

#k = int(input("input neighbors: "))
k = 10

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

#WEEK 02 CONTENT

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

#WEEK 03 CONTENT

df = pd.read_table('car.data.txt', sep=',')
df2 = pd.read_table('pima-indians-diabetes.data.txt', sep=',')
df3 = pd.read_table('auto-mpg.data.txt', sep='\s+')

fullData = df.values
fullData2 = df2.values
fullData3 = df3.values


# Data Preparation
carTarget, carData = carDataPrep(fullData)
indianTarget, indianData = indianDataPrep(fullData2)
fullData3 = mpgDataPrep(fullData3)

# Car Results
car_train1, car_test1, car_train2, car_test2 = train_test_split(carData, carTarget, test_size=0.30)
classifier4 = HardCodedClassifier(k)
model4 = classifier4.fit(car_train1, car_train2)

predictions4 = model4.predict(car_test1)

match4 = 0
for i in range(len(car_test2)):
   one = car_test2[i]
   two = predictions4[i]
   if one == two:
       match4 += 1

# Indian Results

indian_train1, indian_test1, indian_train2, indian_test2 = train_test_split(indianData, indianTarget, test_size=0.30)
classifier5 = HardCodedClassifier(k)
model5 = classifier5.fit(indian_train1, indian_train2)

predictions5 = model5.predict(indian_test1)


match5 = 0
for i in range(len(indian_test2)):
   one = indian_test2[i]
   two = predictions5[i]
   if one == two:
       match5 += 1


#RESULTS
print("IRIS")
print("SkLearn: " + str(match1) + "/" + str(len(iris_test2)))
print("GaussianNB: " + str(match2) + "/" + str(len(iris_test2)))
print("Hardcoded: " + str(match3) + "/" + str(len(iris_test2)))

print("OTHER DATA SETS")
print("Car: " + str(match4) + "/" + str(len(car_test2)))
print("Indian: " + str(match5) + "/" + str(len(indian_test2)))
