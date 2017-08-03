#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 19:13:01 2017

@author: yangfan
"""
from sklearn import datasets
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
def compute_cost(x, y, theta):
    m = np.shape(x)[0]
    pred = np.dot(x, theta.T)
    cost = sum((pred-y)**2)/(2*m)
    return cost
def gradient_descent(x, y, learning_rate, num_iterations, threshold):
    m = np.shape(x)[0] #
    n = np.shape(x)[1]
    theta = np.zeros(shape=(1, n))
    for j in xrange(num_iterations):
        predictions = np.dot(x, theta.T)
        
        y=np.reshape(y,np.shape(predictions))
        delta = np.dot((predictions - np.array(y)).T, np.array(x))/m
        theta = theta - learning_rate * delta;
        cost = compute_cost(x,y,theta)
        if cost < threshold:
            break
    return theta


class LinerRegression():
    def __init__(self, NUM_ITERATIONS=10, LEARNING_RATE = 0.01, normalize = False, threshold = 1e-7):
        self.NUM_ITERATIONS = NUM_ITERATIONS
        self.LEARNING_RATE = LEARNING_RATE
        self.normalize = normalize
        self.threshold = threshold
    def fit(self, x, y):
        if self.normalize == True:
            x = preprocessing.scale(x) #feature_normalize
        x = np.append(np.ones([np.shape(x)[0], 1]), x, 1)
        self.theta = gradient_descent(x, y, self.LEARNING_RATE, self.NUM_ITERATIONS, self.threshold)
    def predict(self, x):
        x = np.append(np.ones([np.shape(x)[0], 1]), x, 1)
        return np.dot(x, self.theta.T)
    def save(self):
        self.b = self.theta[0][0]
        self.w = self.theta[0][1:]
        return self.w, self.b
    def test(self, x):
        return np.dot(w, x) + b
    
x,y = datasets.make_regression(n_samples=100, n_features=15, noise = 10)

clf = LinerRegression(LEARNING_RATE=0.5)
clf.fit(x,y)
predict_y = clf.predict(x)
w, b = clf.save()

#plt.scatter(x, y, color='black')
#plt.plot(x,predict_y,color='red')
#plt.show()
