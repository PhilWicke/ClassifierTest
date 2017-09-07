# -*- coding: utf-8 -*-

from sklearn.naive_bayes import GaussianNB
from sklearn import svm
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
print("GAUSSIAN NAIVE BAYES:")
print("Success rate of Gaussian Naive Bayes is:\t %.3f \n" % (successRate))

# SUPPORT VECTOR MACHINE
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

