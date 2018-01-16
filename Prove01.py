####################################################
# Prove 01, Experiment Shell & Hardcoded Classifier
# Author: Jon Crawford
# Professor: Brother Burton
# Summary - Some beginner stuffs for class
####################################################

import sys

from sklearn import datasets
from sklearn import utils
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

# Classifier itself
class HardCodedClassifier():
    def fit(self, trainData, trainTarget):
       model = Model(0)
       return model
    
# Mode to user for the classifier
class Model():
    def __init__(self, number):
       self.number = number
    
    def predict(self, testData):
       return 0

# Load the data
iris = datasets.load_iris()

# Randomize split into a training set: (70%) and testing set (30%)
iris_train1, iris_test1, iris_train2, iris_test2 = train_test_split(iris.data, iris.target, test_size=0.30)

# Create model (GaussianNB)
classifier = GaussianNB()
model = classifier.fit(iris_train1, iris_train2)

# Create model (Hardcoded)
classifier2 = HardCodedClassifier()
model2 = classifier2.fit(iris_train1, iris_test1)

# Make predictions
targets_predicted = model.predict(iris_test1)

# Gaussian Predictions
match1 = 0
for i in range(len(iris_test2)):
   one = iris_test2[i]
   two = targets_predicted[i]
   if one == two:
      match1 += 1

# Hardcoded Predictions
match2 = 0
for i in range(len(iris_test2)):
   one = iris_test2[i]
   two = model2.predict(iris_test2[i])
   if one == two:
       match2 += 1

#Present accuracy percentages
print("GaussianNB: " + str(match1) + "/" + str(len(iris_test2)))
print("Hardcoded: " + str(match2) + "/" + str(len(iris_test2)))
