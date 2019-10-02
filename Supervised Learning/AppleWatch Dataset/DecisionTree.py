# Author: Abu Kaisar Jamil , University of Asia Pacific

#Platform: Google Colabotary

#import pandas and numpy
import pandas as pd
import numpy as np

#load csv file of the dataset
applewatch = pd.read_csv('AppleWatch.csv')

# print the shape and summary of the dataset
print(applewatch.shape)
print(applewatch.head())

# preprocessing the features and labels
train = applewatch.drop('Heart',axis=1)
y= applewatch["Heart"]
x=train.drop(['Activity','Gender'], axis=1)
total = [x,y]


# applying decision tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=.20,random_state=1)
x_train.shape, y_train.shape

clf = DecisionTreeClassifier(random_state=1)
clf.fit(x_train, y_train)

predictions = clf.predict(x_test)

print(predictions)
