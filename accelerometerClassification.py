#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 19:59:41 2018

@author: Yuliy Tsank
"""

from __future__ import print_function
import argparse

from sklearn import preprocessing, linear_model
import os
import csv
import numpy as np
import random
from matplotlib import pyplot, rc
from matplotlib.legend_handler import HandlerLine2D
from pylab import savefig

'''
Training settings as optional parameters with default values
'''
parser = argparse.ArgumentParser(description='Accelerometer Data Classification')
parser.add_argument('--test-prop', type=int, default=0.2, metavar='TP',
                    help='learning rate (default: 0.01)')
parser.add_argument('--batch-size', type=int, default=50, metavar='BS',
                    help='input batch size for training and testing (default: 50)')
parser.add_argument('--epochs', type=int, default=50, metavar='E',
                    help='number of epochs to train (default: 50)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--csv-file-path', type=str, default='dataset-har-PUC-Rio-ugulino.csv', metavar='CSV',
                    help='csv file path (default: dataset-har-PUC-Rio-ugulino.csv)')
args = parser.parse_args()

'''
####Extract parameters settings from arguments#################################
'''
batch_size = args.batch_size            #batch sizes to use for training and testing
num_epochs = args.epochs                #number of epochs to run full data through 
path_to_csv = args.csv_file_path        #path to csv file 
test_prop = args.test_prop              #proption of data to use for testing
every_nth = round(10/(test_prop*10.))   #take every nth sample for testing based on proportion of data to use for testing

'''
Extract data from csv file line by line
'''
csv_lines = []                                          #preallocate list for extracted csv lines
with open(path_to_csv, 'rb') as csvfile:
     rowreader = csv.reader(csvfile, delimiter=';')
     next(rowreader)                                    #ignore first header line

     for row_num, row in enumerate(rowreader):
         csv_lines.append(row)                          #append row info
random.shuffle(csv_lines)                               #shuffle data 

classes = ('sitting', 'sittingdown', 'standing', 'standingup', 'walking' ) # record names of classes to assign integers to them later

'''
####Preallocate dictionary lists for training and testing data to be split into batches
'''
all_input_data = {}
all_target_data = {}
all_input_data['test'] = []
all_input_data['train'] = []
all_target_data['test'] = []
all_target_data['train'] = []

batch_data = {}
batch_target = {}
batch_data['test'] = []
batch_target['test'] = []
batch_data['train'] = []
batch_target['train'] = []

counters = {}
counters['train'] = 0;                # set counter for number of training data samples
counters['test'] = 0                  # set counter for number of testing data samples

# Instantiate objects for logistic regression model and support vector machine model
logit_model = linear_model.SGDClassifier(loss = 'log', warm_start=True)
svm_model = linear_model.SGDClassifier(loss='hinge', warm_start=True)

'''
####Split data into training and testing sets based on a set proportion. Then split
####each of them into batches
'''
scaler = preprocessing.RobustScaler() #Instantiate object for data scaler

for row_num, row in enumerate(csv_lines): # Loop through each row of the data extracted from csv file
     
     row = [s.replace(',', '.') for s in row]   #replace commas with dots for decimal places to convert from European convention
     if row[1] == 'Woman':                      #replace recorded genders with integers
         row[1] = 0
     else:
         row[1] = 1
     row[18] = classes.index(row[18])           #replace activity labels with integers based on list above
     del(row[0])                                #delete subject identifier so that all data is treated as one large set
     row_data = np.array(row)                   #convert row to a numpy array with integer values 
     
     try:
         float_row_data = row_data.astype(np.float) #convert integers to float values unless there is missing data
     except:
         continue
                            
     if (row_num % every_nth == 0):             #deterimine whether data in this row will be used for trianing or testing
         data_type = 'test'
         counters['test']+=1
     else:
         data_type = 'train'
         counters['train']+=1
         
     batch_data[data_type].append(float_row_data[0:17])                 #place row feature data into current batch depending on whether its train or test data 
     batch_target[data_type].append(float_row_data[17].astype(np.long)) #place row label data into current batch depending on whether its train or test data 
              
     #if batch is full for current test batch or train batch, then place it into a list and start a new batch
     if counters[data_type] == batch_size:
       
        batch_data_test = np.concatenate((np.array(batch_data[data_type]),      #concatente test features and labels to shuffle them together
                                          np.array(batch_target[data_type])[:,None]), 1)
        
        np.random.shuffle(batch_data_test)                                      #shuffle test data in case it wasn't shuffled earlier
        batch_data[data_type] = batch_data_test[:, :-1]                         #separate test features from labels
        batch_data[data_type] = scaler.fit_transform(batch_data[data_type])     #scale features of test data to have zero mean and unit variance
        batch_target[data_type] = batch_data_test[:, -1]
         
        all_input_data[data_type].append(batch_data[data_type])                 #append test features batch to list of batches
        all_target_data[data_type].append(batch_target[data_type])              #append test target batch to list of batches
        
        batch_data[data_type] = []                                              #start new batch
        batch_target[data_type] = []
        
        counters[data_type] = 0                                                 #reset batch counter
        
        
'''
####Define traing and testing functions########################################
'''
def train(epoch):
    
    correct_logit = 0
    correct_svm = 0
    for batch_idx, data in enumerate(input_data['train']):                      # loop through list of train batches

        target = np.array(target_data['train'][batch_idx])                      #extract target 
        logit_model.partial_fit(data, target, classes = np.array([0,1,2,3,4]))  #update model parameters using current batch
        svm_model.partial_fit(data,target, classes = np.array([0,1,2,3,4]))

        acc_logit = logit_model.score(data, target)                             #calculate accuracy for logit
        acc_svm = svm_model.score(data, target)                                 #calculate accuracy for svm 
        correct_logit += acc_logit
        correct_svm += acc_svm

        print('Batch Num: ' + str(batch_idx) +' Logit Accuracy: {}, SVM Accuracy: {}'.format(acc_logit, acc_svm))
    
    correct_logit /= len(target_data['train'])
    correct_svm /= len(target_data['train'])
    return correct_logit, correct_svm

def test(epoch):

    test_loss = 0
    correct_logit = 0
    correct_svm = 0
    for batch_idx, data in enumerate(input_data['test']):                       #loop through list of test batches

        target = target_data['test'][batch_idx]                                 #extract target
        acc_logit = logit_model.score(data, target)                             #update model parameters using current batch
        acc_svm = svm_model.score(data, target)
        correct_logit += acc_logit                                              #calculate accuracy for logit
        correct_svm += acc_svm                                                  #calculate accuracy for svm

    correct_logit /= len(target_data['test'])
    correct_svm /= len(target_data['test'])
    test_loss = test_loss
    test_loss /= len(target_data['test']) # loss function already averages over batch size
    print('Test: Logit Accuracy: {}, SVM Accuracy: {}'.format(correct_logit, correct_svm))
    return correct_logit, correct_svm

'''
####Set parameters for error analysis of train set sizes and regularizer values 
'''

train_props = np.linspace(0.01,1,20)                                            #Proportions of training set to test for error analysis
num_train_batches = len(all_input_data['train'])
train_inds = np.round(train_props*num_train_batches).astype(int)                #Find indices to extract train sets of different sizes
num_train_inds = len(train_inds)
regularizer_vals= [1E-4, .001, .01, .1]                                         #Set the regularizer values to try
num_regularizer_vals = len(regularizer_vals)

# Preallocate space to record output of error analysis
error_analysis ={}
error_analysis['train_size'] = {}
error_analysis['reg_val'] = {}
error_analysis['train_size']['train'] = np.empty((num_epochs,num_train_inds,2))*np.nan
error_analysis['train_size']['test'] = np.empty((num_epochs,num_train_inds,2))*np.nan
error_analysis['reg_val']['train'] = np.empty((num_epochs,num_regularizer_vals,2))*np.nan
error_analysis['reg_val']['test'] = np.empty((num_epochs,num_regularizer_vals,2))*np.nan

'''
####Test effects of training set sizes on performance##########################
'''

for train_size_idx, train_size in enumerate(train_inds):
    
    input_data = {}
    target_data = {}
    input_data['train'] = all_input_data['train'][0:train_inds[train_size_idx]] # Extract a training set of a particular size depending on current index
    target_data['train'] = all_target_data['train'][0:train_inds[train_size_idx]]
    input_data['test'] = all_input_data['test']
    target_data['test'] = all_target_data['test']
    
    # Instantiate objects for logistic regression model and support vector machine model
    logit_model = linear_model.SGDClassifier(loss = 'log', alpha=1E-4, warm_start=True) #need to refit model again for each training set size
    svm_model = linear_model.SGDClassifier(loss='hinge', alpha=1E-4, warm_start=True)
    
    for epoch in range(0,num_epochs):                                            #run each training set size through multiple epochs of training 
        
        correct_logit, correct_svm = train(epoch)
        error_analysis['train_size']['train'][epoch, train_size_idx,0] = correct_logit  #record performance of training set
        error_analysis['train_size']['train'][epoch, train_size_idx,1] = correct_svm
        correct_logit, correct_svm = test(epoch)
        error_analysis['train_size']['test'][epoch, train_size_idx,0] = correct_logit   #record performance of testing set
        error_analysis['train_size']['test'][epoch, train_size_idx,1] = correct_svm
        
        np.save('error_analysis', error_analysis )
'''
####Plot performance as a function of training set size########################
'''
label1 = 'Logit'
label2 = 'SVM'
lwidth = 3
axis_fontsize = 40
title_fontsize = 50
rc('xtick', labelsize=25) 
rc('ytick', labelsize=25) 
pyplot.figure(num=None, figsize=(10, 8), dpi=80)

# Plot separate lines for testing and training sets for logit and svm models (4 lines total)
line1, = pyplot.plot(train_inds, error_analysis['train_size']['train'][-1,:,0], 'b--', 
                     linewidth=lwidth, label = 'Train Set -' +label1)
line2, = pyplot.plot(train_inds,error_analysis['train_size']['test'][-1,:,0], 'b', 
                     linewidth=lwidth, label = 'Test Set -' +label1)
line1, = pyplot.plot(train_inds, error_analysis['train_size']['train'][-1,:,1], 'r--', 
                     linewidth=lwidth, label = 'Train Set -'+label2 )
line2, = pyplot.plot(train_inds, error_analysis['train_size']['test'][-1,:,1], 'r', 
                     linewidth=lwidth, label = 'Test Set -'+label2)
pyplot.ylim(.75, .85)
pyplot.xlabel('Train Set Size (batches of ' + str(batch_size) + ')', fontsize = axis_fontsize)
pyplot.ylabel('Proportion Correct', fontsize = axis_fontsize)
pyplot.title('Performance', fontsize = title_fontsize)

pyplot.legend(fontsize = 20, handler_map={line1: HandlerLine2D()}, loc='center left',
                           bbox_to_anchor=(1, 0.5), bbox_transform=pyplot.gcf().transFigure)
savefig('TrainSizeEffects.png', bbox_inches='tight')
pyplot.show()

'''
####Test effects of regularizer values on performance##########################
'''

for reg_val_idx, reg_val in enumerate(regularizer_vals):
    
    input_data = {}
    target_data = {}
    input_data['train'] = all_input_data['train']
    target_data['train'] = all_target_data['train']
    input_data['test'] = all_input_data['test']
    target_data['test'] = all_target_data['test']
    
    # Instantiate objects for logistic regression model and support vector machine model
    logit_model = linear_model.SGDClassifier(loss = 'log', alpha=reg_val, warm_start=True)
    svm_model = linear_model.SGDClassifier(loss='hinge', alpha=reg_val, warm_start=True)
    for epoch in range(0,num_epochs):
        
        correct_logit, correct_svm = train(epoch)
        
        error_analysis['reg_val']['train'][epoch, reg_val_idx,0] = correct_logit
        error_analysis['reg_val']['train'][epoch, reg_val_idx,1] = correct_svm

        correct_logit, correct_svm = test(epoch)
        error_analysis['reg_val']['test'][epoch, reg_val_idx,0] = correct_logit
        error_analysis['reg_val']['test'][epoch, reg_val_idx,1] = correct_svm
        
        np.save('error_analysis', error_analysis )
 
    ####Plot performance as a function of training set size for current regularizer value
    epoch_range = range(0,num_epochs)
    pyplot.figure(num=None, figsize=(10, 8), dpi=80)
    
    # Plot separate lines for testing and training sets for logit and svm models (4 lines total)
    line1, = pyplot.plot(epoch_range, error_analysis['reg_val']['train'][:,reg_val_idx,0], 'b--', 
                         linewidth=lwidth, label = 'Train Set -' +label1)
    line2, = pyplot.plot(epoch_range,error_analysis['reg_val']['test'][:,reg_val_idx,0], 'b', 
                         linewidth=lwidth, label = 'Test Set -' +label1)
    line1, = pyplot.plot(epoch_range, error_analysis['reg_val']['train'][:,reg_val_idx,1], 'r--', 
                         linewidth=lwidth, label = 'Train Set -'+label2 )
    line2, = pyplot.plot(epoch_range, error_analysis['reg_val']['test'][:,reg_val_idx,1], 'r', 
                         linewidth=lwidth, label = 'Test Set -'+label2)
    pyplot.ylim(.75, .85)
    pyplot.xlabel('Epoch Number', fontsize = axis_fontsize)
    pyplot.ylabel('Proportion Correct', fontsize = axis_fontsize)
    pyplot.title('Regularizer: ' + str(regularizer_vals[reg_val_idx]), fontsize = title_fontsize)
    
    pyplot.legend(fontsize = 20, handler_map={line1: HandlerLine2D()}, loc='center left',
                           bbox_to_anchor=(1, 0.5), bbox_transform=pyplot.gcf().transFigure)
    savefig('RegularizerEffects' + str(reg_val_idx) + '.png', bbox_inches='tight')
