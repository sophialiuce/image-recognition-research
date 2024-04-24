#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 07:23:35 2018

@author: sophialiu
"""

import sklearn.svm as ssv  
from sklearn.externals import joblib   
import time 
import numpy as np 
from sklearn.model_selection import train_test_split
  
if __name__ == "__main__":  
    model_path = './models/svm_brand.model'  
    x_features = np.load('hog_features.npy')
    y_product = np.load('train_data_y_brand_224.npy')
    
    X, test_X, Y, test_Y = train_test_split(x_features, y_product, 
                                        random_state=1, 
                                        train_size=0.65)
    
    t0 = time.time()
#------------------------SVM--------------------------------------------------  
    clf = ssv.SVC(C=1, kernel='linear', decision_function_shape='ovo')
    print ("Training a SVM Classifier.")
    clf.fit(X, Y)  
    joblib.dump(clf, model_path)
    
    print ("The accuracy of the training set is: ", clf.score(X, Y))  # Accuracy
    y_hat = clf.predict(X)
#    show_accuracy(y_hat, Y, 'Train')
    
    print ("The accuracy of the testing  set is: ", clf.score(test_X, test_Y))
    y_hat_b = clf.predict(test_X)
#    show_accuracy(y_hat, test_Y, 'Test')
#------------------------SVM--------------------------------------------------  
    t1 = time.time()  
    print ("Classifier saved to {}".format(model_path))
    print ('The cast of time is :%f seconds' % (t1-t0))