#Classifiers -*- coding: utf-8 -*-
"""
Created on Sat Dec 22 17:02:47 2018

@author: WraithDelta
"""
from sklearn import tree #decision tree model of scikit
from sklearn import dummy
from sklearn import neural_network


#define variable clf to store decision tree classifier
clf = tree.DecisionTreeClassifier()  # reference tree dependency by call decision tree method on tree object
dclf = dummy.DummyClassifier()
nrl = neural_network.MLPClassifier() 



#[height, weight, shoe size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
    [190, 90, 47], [175, 64, 39],
    [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']

# Challenge train them on data"

clf = clf.fit (X, Y)  # call fit method on classifier variable that takes two args
#fit method trains the decision tree on our data set
dclf = dclf.fit (X, Y)

nrl = nrl.fit (X, Y)

#define variable prediction to store results 
prediction = clf.predict([[190, 70, 39]]) #call predict method given 3 values
prediction2 = dclf.predict([[190, 70, 39]])
prediction3 = nrl.predict([[190, 70, 39]])

#Challenge compare their results and print the best one!


print (prediction)
print (prediction2) 
print (prediction3) 

