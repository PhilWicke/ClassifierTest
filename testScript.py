# -*- coding: utf-8 -*-

from sklearn.naive_bayes import GaussianNB
import numpy as np

# Enter filePath
filePath = './testFileN10x1000.txt'

# Processing tab-separated file
table = np.loadtxt(filePath, delimiter='\t')

# Splitting data and target values
data = table[:,:-1]
targ = table[:,-1]
# Converting output into actual integer values
targ = list(map(int, targ))


# GAUSSIAN NAIVE BAYES
gnb = GaussianNB()
y_pred = gnb.fit(data, targ).predict(data)
successRate = 1-(1/data.shape[0])*(targ != y_pred).sum()
print("Success rate of Gaussian Naive Bayes is: %.3f " % (successRate))
