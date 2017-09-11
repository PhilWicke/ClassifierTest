'''
This file contains the scripts to test different classifiers with tab-
separated files in a modular structure of a class.
'''
import os.path
import numpy as np
from sklearn.externals import joblib
from sklearn.naive_bayes import GaussianNB


class testerClass:
    
    def __init__(self):
        # class variables
        self.data        = []
        self.target      = []
        self.model       = 0;
        self.pathFile    = ""
    
    '''
    Loads the dataset from the filepath into two data fields of data and
    target arrays. Raises errors if file is non-existent or data not in
    a useful format after readout.
    '''
    def load_dataset(self,filepath):
        if os.path.exists(filepath):
            # Processing tab-separated file
            table = np.loadtxt(filepath, delimiter='\t')
            
        # Error handling
            if (np.shape(table)[0] < 1) or (np.shape(table)[1]<2):
                raise ValueError('Text file resulted in bad data.')
        else:
            raise ValueError('File not found.')
        
        # Splitting data and target values
        self.data = table[:,:-1]
        target = table[:,-1]
        # Converting output into actual integer values
        self.target = list(map(int, target))
    
    '''
    Naive Gaussian Bayes Classifier:
    GNB algorithm for classification. The likelihood of the features
    is assumed to be Gaussian.    
    '''
    def train_GNB(self):
        gnb = GaussianNB()
        self.model = gnb.fit(self.data, self.target)
        y_pred = self.model.predict(self.data)
        successRate = 1-(1/self.data.shape[0])*(self.target != y_pred).sum()
        print("Success rate of Gaussian Naive Bayes is:\t %.3f \n" % (successRate))
        
    def save_model(self,outputfilepath):
        # Create model file        
        filename = outputfilepath+'.joblib.pkl'
        joblib.dump(self.model, filename, compress=9)
        