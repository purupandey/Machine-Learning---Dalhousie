#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 11:51:06 2017

@author: tt
"""

from pylab import *
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
from tensorflow.contrib import learn
tf.logging.set_verbosity(tf.logging.ERROR)



# data (generation of training data)
n =100
x1= array([randn(n)+1,randn(n)+1]);y1=zeros(n,int)
x2= array([randn(n)+3,randn(n)+3]);y2=zeros(n,int)+1
x = hstack((x1,x2)).T
y = hstack((y1,y2))
plot(x1[0,:] , x1[1,:] , 'xr' )
plot(x2[0,:] , x2[1,:] , 'ob' )
show()


# making model and training (fitting) them
SVC = svm.SVC(kernel ='linear', C=1)
SVC.fit(x,y)

RF = RandomForestClassifier(n_estimators=10)
RF.fit(x,y)

feature_columns = [tf.contrib.layers.real_valued_column("", dimension=4)]
MLP = learn.DNNClassifier(
        feature_columns=feature_columns,
        hidden_units = [10, 20, 10],
        n_classes = 2)
MLP.fit(x,y,steps = 500, batch_size = 128)

# generating testing data
x1= array([randn(n)+1,randn(n)+1]);y1= zeros(n)
x2= array([randn(n)+3,randn(n)+3]);y2= zeros(n)+1

x = hstack((x1 ,x2)).T
y = hstack((y1 ,y2))

# prediction
a=SVC.predict(x)
b=RF.predict(x)
c=list(MLP.predict(x,as_iterable=True))

#evaluation
print('PercentageCorrect SVM: ', (n-sum(abs(y-a)))/n )
print('PercentageCorrect RF : ', (n-sum(abs(y-b)))/n )
print('PercentageCorrect MLP : ', (n-sum(abs(y-c)))/n )
# =============================================================================
