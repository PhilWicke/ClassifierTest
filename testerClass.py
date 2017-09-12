'''
This file contains the scripts to test different classifiers with tab-
separated files in a modular structure of a class.
'''
import os.path
import numpy as np
from sklearn.externals import joblib
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.metrics import confusion_matrix


class testerClass:
    
    '''
    Initializing the class with four variables for data, target, model and
    filepath.
    '''
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
        
        #y_pred = self.model.predict(self.data)
        #successRate = 1-(1/self.data.shape[0])*(self.target != y_pred).sum()
        #print("Success rate of Gaussian Naive Bayes is:\t %.3f \n" % (successRate))
        
    '''
    Support Vector Machine:
        C       : Penalty parameter C of the error term.
        kernel  : string, optional (default=’rbf’)
                  Specifies the kernel type to be used in the algorithm.
                  ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’ or a callable.
        degree  : Degree of the polynomial kernel function (‘poly’).
        gamma   : Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’. 
    '''
    def train_SVC(self,kernel="rbf",C=1.0,gamma=0.7,degree=3):
        self.model = svm.SVC(kernel=kernel,C=C,gamma=gamma,degree=degree)
        self.model = self.model.fit(self.data, self.target)
        
        #cross_val_score = self.model.score(self.data,self.target)
        #print("Success rate of a SVC w/ linear kernel is:\t %.3f " % (cross_val_score))
        
    # Linear Kernel Version of SVM
    def train_linearSVC(self,C=1.0):
        self.model = svm.LinearSVC(C=C)
        self.model = self.model.fit(self.data, self.target)
        
    '''
    Function that returns a model prediction
    '''
    def predict(self,data):
        return self.model.predict(data)
        
    """
    Function to compute confusion matrix to evaluate the accuracy of a classification
    """
    def get_confusion_matrix(self,data,target):
        y_pred = self.predict(data)
        return confusion_matrix(target,y_pred)
        
    '''
    Function to save model. 
    String - outputfilepath will be extended by *.joblib.pkl to save it as
             pickle archive 
    '''
    def save_model(self,outputfilepath):
        # Create model file        
        filename = outputfilepath+'.joblib.pkl'
        joblib.dump(self.model, filename, compress=9)
        
    '''
    Function to load a model. 
    String - modelfilepath must have extension of *.joblib.pkl to load it as
             pickle archive
    '''
    def load_model(self,modelfilepath):
        # Load model from file
        if '\.joblib.pkl' not in modelfilepath:
            self.model = joblib.load(modelfilepath)
        else:
            raise ValueError("Provided model path incorrect (*.joblib.pkl)")