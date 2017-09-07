# -*- coding: utf-8 -*-
'''
Script to transform the iris dataset into a txt file, so it can be used
by the sampleGenerator.py, which can only process tab separated txt files.
'''
from sklearn import datasets
iris = datasets.load_iris()

filePath = './irisFile.txt'
n = iris.data.shape[0] # 150
m = iris.data.shape[1] # 4


with open(filePath, 'w') as f:
    for i in range(n):
        for j in range(m):
            # set range of floats for data
            f.write(str(iris.data[i-1,j-1])+'\t')
        # set range of ints for data
        f.write(str(iris.target[i])+'\n')  