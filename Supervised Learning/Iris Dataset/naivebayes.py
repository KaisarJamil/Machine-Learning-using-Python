# Author: Abu Kaisar Jamil , University of Asia Pacific

#Platform: Google Colabotary

#import pandas and numpy
import pandas as pd
import numpy as np


#load csv file of the dataset
data = pd.read_csv('iris.csv')

# print the shape and summary of the dataset
print(data.shape)
print(data.head())

# preprocessing the features and labels
x = data.drop('Variety',axis=1)
y = data['Variety']
X = data.drop('Variety',axis=1)

# testing with 20% data for 80% data's
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.20)

#applying naive bayes algorithm
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(y_pred)
