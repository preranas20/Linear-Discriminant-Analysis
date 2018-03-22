#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 10:43:50 2017
@author: preranasingh
"""

############


import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
import pandas as pd
import matplotlib.lines as mlines



#Caculating the within_class scatter matrix for each class
def calculate_scatter_within(mean_data):
    scatter_within = np.zeros((19,19))
    for data_class,mean_value in zip(range(1,3), mean_data):
       class_scatter_mat = np.zeros((19,19))                 
       for val in X[y == data_class]:
          val, mean_value = val.reshape(19,1), mean_value.reshape(19,1) 
          class_scatter_mat = class_scatter_mat+(val-mean_value).dot((val-mean_value).T)
       scatter_within =scatter_within+class_scatter_mat                      
    return scatter_within    
    

#calcualting the scatter matrix between the classes    
def calculate_between_class_scatter(total_mean,mean_data):
    scatter_between = np.zeros((19,19))
    for i,mean_value in enumerate(mean_data):  
        n = X[y==i+1,:].shape[0]  
        mean_value = mean_value.reshape(19,1) 
        total_mean = total_mean.reshape(19,1) 
        scatter_between =scatter_between+ n * (mean_value - total_mean).dot((mean_value - total_mean).T)
    return scatter_between
  

    
#Performing LDA Algorithm on the data
def lda(X,y):
    for data_class in range(1,3):
       mean_data.append(np.mean(X[y==data_class], axis=0))
    
    scatter_within=calculate_scatter_within(mean_data)
    
    total_mean = np.mean(X, axis=0)
    total_mean
    
    scatter_between=calculate_between_class_scatter(total_mean,mean_data)
    
    eigen_values, eigen_vectors = np.linalg.eig(np.linalg.inv(scatter_within).dot(scatter_between))

    
    eigen_pairs = [(np.abs(eigen_values[i]), eigen_vectors[:,i]) for i in range(len(eigen_values))]  

    eigen_pairs_sorted = sorted(eigen_pairs, key=lambda k: k[0], reverse=True)
    #As we can see after sorting the eigen values that only one of teh eigen value is giving giving the highest values
    #LDA gives 1 component after applying the algorithm
    
    comb_mat = np.hstack((eigen_pairs_sorted[0][1].reshape(19,1),eigen_pairs_sorted[1][1].reshape(19,1)))
    print('Matrix comb_mat:\n', comb_mat.real)
    lda_matrix = X.dot(comb_mat)      
    return lda_matrix


#Read data from file 
in_file_name = "/Users/preranasingh/Documents/sem2/ML/Quiz and  Exam-sample/Homework 2/SCLC_study_output_filtered_2.csv"
data_in = pd.read_csv(in_file_name, index_col=0)
data_in
mean_data = []

X = data_in.as_matrix()
X
y = np.concatenate((np.ones(20), np.ones(20)+1))
y=y.astype(numpy.int64)
y

#Result of applying LDA Algorithm to the input
lda_matrix=lda(X,y)


#Plotting graphs after applying LDA algorithm on the data
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_title('Results from applying LDA on data')
ax.set_xlabel('LD1')
ax.set_ylabel('LD2')
ax.plot(lda_matrix[0:19,0], np.zeros(19), linestyle='None', marker='o', markersize=5, color='blue', label='NSCLC')
ax.plot(lda_matrix[20:39,0], np.zeros(19), linestyle='None', marker='o', markersize=5, color='red', label='SCLC')
ax.legend()


#Comapring the results of the algorithm with that of sklearn.discriminant analysis.LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
sklearn_lda = LDA(n_components=1)
X_lda_sklearn = sklearn_lda.fit_transform(X, y)


#Plotting graphs after applying sklearn.discriminant analysis.LinearDiscriminantAnalysis to the data
fig = plt.figure()
fx = fig.add_subplot(1, 1, 1)
fx.set_title('Results from applying Sklearn.LDA on data')
fx.set_xlabel('LD1')
fx.set_ylabel('LD2')
fx.plot(X_lda_sklearn[0:19,0], np.zeros(19), linestyle='None', marker='*', markersize=5, color='blue', label='NSCLC')
fx.plot(X_lda_sklearn[20:39,0], np.zeros(19), linestyle='None', marker='*', markersize=5, color='red', label='SCLC')
fx.legend()

fig.show()