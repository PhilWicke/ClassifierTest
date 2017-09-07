# -*- coding: utf-8 -*-
'''
This script will allow to take a text file with tab separated data and apply 
different classify algorithms to it. 
'''
from sklearn.naive_bayes import GaussianNB
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import svm
import numpy as np

#filePath = './testFileN10x1000.txt'
filePath  = './irisFile.txt'

# Processing tab-separated file
table = np.loadtxt(filePath, delimiter='\t')

# Splitting data and target values
data = table[:,:-1]
targ = table[:,-1]
# Converting output into actual integer values
targ = list(map(int, targ))


#%% GAUSSIAN NAIVE BAYES
gnb = GaussianNB()
y_pred = gnb.fit(data, targ).predict(data)
successRate = 1-(1/data.shape[0])*(targ != y_pred).sum()
print("GAUSSIAN NAIVE BAYES:")
print("Success rate of Gaussian Naive Bayes is:\t %.3f \n" % (successRate))

#%% SUPPORT VECTOR MACHINE
C = 1.0
models = (svm.SVC(kernel='linear', C=C),
          svm.LinearSVC(C=C),
          svm.SVC(kernel='rbf', gamma=0.7, C=C),
          svm.SVC(kernel='poly', degree=3, C=C))

models = (clf.fit(data, targ) for clf in models)
cross_val_scores = []
for clf in models:
    cross_val_scores.append(clf.score(data,targ))

print("SUPPORT VECTOR MACHINE:")
print("Success rate of a SVC w/ linear kernel is:\t %.3f " % (cross_val_scores[0]))
print("Success rate of a linearSVC is:\t\t\t %.3f " % (cross_val_scores[1]))
print("Success rate of a SVC with RBF is:\t\t %.3f " % (cross_val_scores[2]))
print("Success rate of a SVC w/ polynomial deg. is:\t %.3f \n" % (cross_val_scores[3]))

#%% LINEAR REGRESSION

# Percentage of training data
perc_train = 0.1

perc_test  = 1-perc_train
train_lim  = int(len(data)*perc_train)
# Splitting train and test data
data_train = data[0:train_lim]
data_test  = data[train_lim:]
targ_train = targ[0:train_lim]
targ_test  = targ[train_lim:]
                  
# Create regression object
regr = linear_model.LinearRegression()             
# Perform fitting
fit = regr.fit(data_train,targ_train)
# Do prediction
data_pred = regr.predict(data_test)
# Get prediction error
pred_error = 1-(1/data_test.shape[0])*(targ_test != data_pred).sum() 

print('LINEAR REGRESSION:')
print("Success rate of linear regression is:\t\t %.3f " % (pred_error))
# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(targ_test, data_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(targ_test, data_pred))

