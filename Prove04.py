####################################################
# Prove 04, Decision Tree
# Author: Jon Crawford
# Professor: Brother Burton
# Summary - using decision trees
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

# Classifier itself
class HardCodedClassifier():
    def __init__(self):
        pass
    
    def fit(self, trainData, trainTarget):
       model = Model(trainData, trainTarget)
       return model
    
    def findBranch(self, data, target):
       attributeSet = []
       entrophy = []
       countSet = []
   
       for i in range (len(data[0])):
          miniEntrophy = [] #entrophies
          miniTarget = []   #target values
          miniCount = []    #supplements countInfo
          miniData = []     #attributes
          countInfo = []    #
   
          for j in range(len(target)):
              if target[j] not in miniTarget:
                  miniTarget.append(target[j])
                  miniCount.append(0)
              if data[j][i] not in miniData:
                  miniData.append(data[j][i])
   
          for l in range(len(miniData)):
             for k in range(len(miniTarget)):
                for j in range(len(miniCount)):
                    miniCount[k] = 0
         
                for j in range(len(target)):
                   if miniData[l] == data[j][i]: 
                      if miniTarget[k] == target[j]:
                         miniCount[k] += 1
         
             #print(i, miniTarget, miniData, miniCount)
             #find entrophy
             e = 0
      
             for k in range (len(miniCount)):
                 if (sum(miniCount) != 0 and miniCount[k] != 0):
                    x = float(miniCount[k]/(sum(miniCount)))
                    e = e + (-(x) * (math.log2(x)))
                
             countInfo.append(miniCount.copy())
             miniEntrophy.append(e)
             #print(miniEntrophy)
      
          attributeSet.append((miniEntrophy, miniTarget, countInfo, miniData))
          entrophy.append(sum(miniEntrophy)/len(miniEntrophy))
   
       for i in range(len(attributeSet)):
          print(attributeSet[i])
   
       for i in range(len(attributeSet[entrophy.index(min(entrophy))][0])):
          if (attributeSet[entrophy.index(min(entrophy))][0][i] == 0):
             pass
          else:
             pass
    
# Mode to user for the classifier
class Model():
    def __init__(self, trainData, trainTarget):
       self.trainData = trainData
       self.trainTarget = trainTarget
    
    def predict(self, testData):
       #resulting array of predictions
       pass

def lensDataPrep(dataSet):
    lensTarget = [i[5] for i in dataSet]
    lensData = [i[1:5] for i in dataSet]
    return lensTarget, lensData

#ACTUAL PROGRAM STARTS

# Load the data
iris = datasets.load_iris()

# Randomize split into a training set: (70%) and testing set (30%)
iris_train1, iris_test1, iris_train2, iris_test2 = train_test_split(iris.data, iris.target, test_size=0.30)

# Load the data for others sets
df = pd.read_table('lenses.data.txt', sep='\s+', header=None)
fullData = df.values
lensTarget, lensData = lensDataPrep(fullData)

lens_train1, lens_test1, lens_train2, lens_test2 = train_test_split(lensData, lensTarget, test_size=0.30)

#go through the training set
classifier = HardCodedClassifier()
classifier.findBranch(lensData, lensTarget)

             
# loop through each individual category
# find entropy for each individual category
# combine and average entropy for each set
# compare all entropies to find best one

'''
If all examples have the same label
    return a leaf with that label
Else if there are no features left to test
    return a leaf with the most common label
Else
    Consider each available feature
    Choose the one that maximizes information gain
    Create a new node for that feature

    For each possible value of the feature
        Create a branch for this value
        Create a subset of the examples for each branch
        Recursively call the function to create a new node at that branch
'''



