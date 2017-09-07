'''
Script to generate a data set for the test of classifiers.
The data consists of tab-separated float types. The last column holds the
output values as integer types. 
'''

import random

# Number of columns (w/o target)
n = 10
# Number of rows
m = 1000
# File name and location
filePath = './testFileN'+str(n)+'x'+str(m)+'.txt'

with open(filePath, 'w') as f:
    for i in range(m):
        for j in range(n):
            # set range of floats for data
            f.write(str(random.uniform(0,1))+'\t')
        # set range of ints for data
        f.write(str(random.randint(0,5))+'\n')   