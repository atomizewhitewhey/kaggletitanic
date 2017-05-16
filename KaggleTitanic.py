#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  7 00:35:05 2017

@author: matthewyeozhiwei
"""

import pandas as pd
import matplotlib 
import matplotlib.pyplot as plt
import seaborn as sns
import math

train = pd.read_csv('/Users/matthewyeozhiwei/Downloads/train.csv')

## 1. Data Cleaning

## First, drop duplicates and useless features
## Name is useless. Cabin number has far too many nans. Ticket number has 681 unique values as such, 
## it should have little to no effect

train = train.drop_duplicates(subset = ['Name', 'PassengerId'])
train = train.drop(labels = ['Cabin','Name','Ticket'], axis = 1) 
## train['Ticket'] = pd.to_numeric(train['Ticket'], errors = 'coerce') 

## Next, categorical features are mapped. Binary features are mapped, ordinal categories are dummied.

train.Sex = train.Sex.map({'male':0, 'female':1})
## train = pd.get_dummies(train, columns = ['Embarked'])

## There are still 177 nan values in the age column. 
## So, I filled it with the means of the Age
train.Age = train.Age.fillna(train.Age.mean())

##print(train.dtypes)
##print(train.head())
##print(train.describe()) I'm confused why the minimum age is 0.42 lol

## 2. Data Visualization
## Most of the data features are categorical so,  conditional histograms/barplots would be best

## i wanted to do a cond bar plot instead. but i dont know why it keeps fritzing, i replaced plt.hist with plt.bar
## and plot_col with plot_col.value_counts()
def cond_hists(df, plot_col, grid_col):
    for col in grid_col:
        grid1 = sns.FacetGrid(df, col=col)
        grid1.map(plt.hist, plot_col, alpha=.7)
    return grid_col
     

grid_col = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked']

## cond_hists(train, 'Survived', grid_col)

## this seems like a better visualization, but its tedious
def cond_bar(grid_col):
    for col in grid_col:
        fig = plt.figure()
        ax = plt.subplot(1,2,1)
        ax.set_title('Passenger did not survive')
        ax.set_ylabel('Counts of '+ col)
        train[train['Survived'] == 0][col].value_counts().plot(kind = 'bar')
        ax = plt.subplot(1,2,2)
        ax.set_title('Passenger survived')
        ax.set_ylabel('Counts of '+ col)
        train[train['Survived'] == 1][col].value_counts().plot(kind = 'bar')

## cond_bar(grid_col)

## you may need to add in a line of code on to rescale the Fare part, the y axis is very long.
def boxplot(df, col, column):
    fig = plt.figure(figsize=(9, 6))
    ax = fig.gca()
    df.boxplot(column = column, by = col, ax = ax)
    ax.set_title('Box plots of ' + column + ' by ' + col)
    ax.set_ylabel(column)
    return column 

##boxplot(train, 'Survived', 'Fare')
##boxplot(train, 'Survived', 'Age')

## Overall conclusions seem to be that :
## People who survive spend more on ship fare
## More females definitely survived. More jacks died.
## Passenger class seem to be in agreement with the fare, where more people in higher class survived
 

## 3. Machine Learning

## before machine learning algorithms are used, ensure all features are numerical!
train = pd.get_dummies(train, columns = ['Embarked'])
## print(train.dtypes)

## Passenger Id is useless and should be irrelevant
train = train.drop(labels = ['PassengerId'], axis = 1)

## Arrange data in to two dataframes of labels and just features

y = train['Survived']
X = train.drop(labels = ['Survived'], axis = 1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 7)
X_test = pd.DataFrame(X_test)

## Data is normalized using Standardisation
## I prefer standardisation over normalisation

from sklearn import preprocessing
stand = preprocessing.StandardScaler()
maxabs = preprocessing.MaxAbsScaler()
minmax = preprocessing.MinMaxScaler()
kernel = preprocessing.KernelCenterer()
normalise = preprocessing.Normalizer()
preprocess = [stand, maxabs, minmax, kernel, normalise]
preprocess_string = ['stand', 'maxabs', 'minmax', 'kernel', 'normalise']

from sklearn import manifold
from sklearn.decomposition import PCA
pca = PCA(n_components = 4)
isomap = manifold.Isomap(n_components = 4, n_neighbors = 7)
dimension = [pca, isomap]
dimension_string = ['pca', 'isomap']

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

rfc = RandomForestClassifier()
abc = AdaBoostClassifier()
gbc = GradientBoostingClassifier()
classifiers = [rfc, abc, gbc]
classifiers_string = ['rfc', 'abc', 'gbc']

## I think pca and isomap dont have to be used and dimensionality 
## reduction should not be an issue
'''
for a in range(0, len(preprocess)):
    trans = preprocess[a]
    trans_s = preprocess_string[a]
    trans.fit(X_train)
    X1_train = trans.transform(X_train)
    X1_test = trans.transform(X_test)
    for b in range(0, len(dimension)):
        reduce = dimension[b]
        reduce_s = dimension_string[b]
        reduce.fit(X1_train)
        X2_train = reduce.transform(X1_train)
        X2_test = reduce.transform(X1_test)
        for c in range(0, len(classifiers)):
            classifier = classifiers[c]
            classifier_s = classifiers_string[c]
            classifier.fit(X2_train, y_train)
            score = classifier.score(X2_test,y_test)
            print('Score: ', score)
            print('Preprocessing: ', trans_s)
            print('Dimensionality Reduction Method: ', reduce_s)
            print('Classifier: ', classifier_s)
'''          
'''        

for a in range(0, len(preprocess)):
    trans = preprocess[a]
    trans_s = preprocess_string[a]
    trans.fit(X_train)
    X1_train = trans.transform(X_train)
    X1_test = trans.transform(X_test)
    for c in range(0, len(classifiers)):
        classifier = classifiers[c]
        classifier_s = classifiers_string[c]
        classifier.fit(X1_train, y_train)
        score = classifier.score(X1_test,y_test)
        print('Score: ', score)
        print('Preprocessing: ', trans_s)
        print('Classifier: ', classifier_s)

'''
## Highest Scores belong to (MinMax, GBC), (MaxAbs, GBC), (Stand, GBC)
## Just use standard scaler, it seems to work best generally
## Run a grid search to find the best hyper parameters

stand.fit(X_train)
X1_train = stand.transform(X_train)
X1_test = stand.transform(X_test)
from sklearn.model_selection import GridSearchCV
parameters = {'n_estimators' : [90,100,110],
                 'max_depth' : [3,4,5]}
optgbc = GridSearchCV(gbc, parameters)
optgbc.fit(X1_train, y_train)
score = optgbc.score(X1_test, y_test)
print("Optimized Classification Score: ", round(score*100, 3))

## No change in optimized score lol, maybe im doing this wrong


test = pd.read_csv('/Users/matthewyeozhiwei/Downloads/test.csv')
test_names = test[['PassengerId', 'Name']]
test = test.drop_duplicates(subset = ['Name', 'PassengerId'])
test = test.drop(labels = ['Cabin','Name','Ticket', 'PassengerId'], axis = 1) 
test.Sex = test.Sex.map({'male':0, 'female':1})
test = pd.get_dummies(test, columns = ['Embarked'])
test.Age = test.Age.fillna(test.Age.mean())
test.Fare = test.Fare.fillna(test.Fare.mean())
print(test[pd.isnull(test).any(axis = 1)])
test_predictions = optgbc.predict(test)
test_predictions = pd.DataFrame(test_predictions)
test_predictions.columns = ['Survival']
results = pd.concat([test_names, test_predictions], axis = 1)
print(results.head())

