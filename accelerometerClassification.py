#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 19:59:41 2018

@author: yuliy
"""

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

import sklearn
from sklearn import preprocessing, linear_model, svm, model_selection
import os
import csv
import numpy as np
import random

from matplotlib import pyplot, rc
from matplotlib.legend_handler import HandlerLine2D
from pylab import savefig



# Training settings
#parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
#parser.add_argument('--batch-size', type=int, default=64, metavar='N',
#                    help='input batch size for training (default: 64)')
#parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
#                    help='input batch size for testing (default: 1000)')
#parser.add_argument('--epochs', type=int, default=100, metavar='N',
#                    help='number of epochs to train (default: 10)')
#parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
#                    help='learning rate (default: 0.01)')
#parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
#                    help='SGD momentum (default: 0.5)')
#parser.add_argument('--no-cuda', action='store_true', default=False,
#                    help='enables CUDA training')
#parser.add_argument('--seed', type=int, default=1, metavar='S',
#                    help='random seed (default: 1)')
#parser.add_argument('--log-interval', type=int, default=10, metavar='N',
#                    help='how many batches to wait before logging training status')
#args = parser.parse_args()
#args.cuda = not args.no_cuda and torch.cuda.is_available()
#
#torch.manual_seed(args.seed)
#if args.cuda:
#    torch.cuda.manual_seed(args.seed)
    
#model_selection.learning_curve...........
#create test and train sets


path_to_csv = os.path.join('.', 'dataset-har-PUC-Rio-ugulino.csv')
csv_lines = []
with open(path_to_csv, 'rb') as csvfile:
     rowreader = csv.reader(csvfile, delimiter=';')
     next(rowreader)

     for row_num, row in enumerate(rowreader):
         csv_lines.append(row)
#fid = open(path_to_csv, "r")
#csv_lines = fid.readlines()
random.shuffle(csv_lines)

#fid.close()
#print(li)

all_input_data = {}
all_target_data = {}


all_input_data['test'] = []
all_input_data['train'] = []
all_target_data['test'] = []
all_target_data['train'] = []

test_batch_num = 1
train_batch_num = 1

elem_num = {}

elem_num['test'] = 0
elem_num['train'] = 0

batch_size = 50

batch_data = {}
batch_target = {}

poly = preprocessing.PolynomialFeatures(2)
scaler = preprocessing.RobustScaler()
logit_model = linear_model.SGDClassifier(loss = 'log', warm_start=True)
svm_model = linear_model.SGDClassifier(loss='hinge', warm_start=True)

classes = ('sitting', 'sittingdown', 'standing', 'standingup', 'walking' )
     
batch_data['test'] = []
batch_target['test'] = []
batch_data['train'] = []
batch_target['train'] = []
train_count = 0;
test_count = 0

test_prop = .2#proption of data to use for testing
every_nth = round(10/(test_prop*10.))#take every nth sample for testing 
 
for row_num, row in enumerate(csv_lines):

     
     row = [s.replace(',', '.') for s in row]
     
     if row[1] == 'Woman':
         row[1] = 0
     else:
         row[1] = 1
         
     row[18] = classes.index(row[18])
     
     del(row[0])#delete subject identifier
        
             
     row_data = np.array(row)
     
     try:
         float_row_data = row_data.astype(np.float)
     except:
         continue
#         tensor_row_data = torch.FloatTensor(float_row_data)
                            
     if (row_num % every_nth == 0):
         data_type = 'test'
         test_count+=1
     else:
         data_type = 'train'
         train_count+=1
         
     
     batch_data[data_type].append(float_row_data[0:17])
     batch_target[data_type].append(float_row_data[17].astype(np.long))
              
     if test_count == batch_size:
       
#         if applied_count >= batch_size/2:
         
        #                     batch_data[data_type] = batch_data[data_type]/np.max(batch_data[data_type],0)
#                     - np.mean(batch_data[data_type],0)

        batch_data_train = np.concatenate((np.array(batch_data['train']), np.array(batch_target['train'])[:,None]), 1)
        batch_data_test = np.concatenate((np.array(batch_data['test']), np.array(batch_target['test'])[:,None]), 1)
        
        np.random.shuffle(batch_data_train)
        np.random.shuffle(batch_data_test)
         
        batch_data['train'] = batch_data_train[:, :-1]
        batch_data['test'] = batch_data_test[:, :-1]
        
        #                     batch_data[data_type] = scaler.fit_transform(poly.fit_transform(batch_data[data_type]))
        
        batch_data['train'] = scaler.fit_transform(batch_data['train'])
        batch_data['test'] = scaler.fit_transform(batch_data['test'])
         
        batch_target['train'] = batch_data_train[:, -1]
        batch_target['test'] = batch_data_test[:, -1]
        
        #imp.fit_transform(batch_data['train'])
        #imp.transform(batch_data['train'])
           
         
        #                     input_data[data_type].append(torch.FloatTensor(batch_data[data_type]))
        #                     target_data[data_type].append(torch.LongTensor(batch_target[data_type]))
         
        all_input_data['train'].append(batch_data['train'])
        all_target_data['train'].append(batch_target['train'])
        
        all_input_data['test'].append(batch_data['test'])
        all_target_data['test'].append(batch_target['test'])
        
        batch_data['test'] = []
        batch_target['test'] = []
        batch_data['train'] = []
        batch_target['train'] = []
        train_count = 0;
        test_count = 0

def train(epoch):
    
    correct_logit = 0
    correct_svm = 0
    for batch_idx, data in enumerate(input_data['train']):
#        if args.cuda:
#            data, target = data.cuda(), target.cuda()
        target = np.array(target_data['train'][batch_idx])
            
        logit_model.partial_fit(data, target, classes = np.array([0,1,2,3,4]))
        svm_model.partial_fit(data,target, classes = np.array([0,1,2,3,4]))

        acc_logit = logit_model.score(data, target)
        acc_svm = svm_model.score(data, target)
        correct_logit += acc_logit
        correct_svm += acc_svm

        print('Batch Num: ' + str(batch_idx) +' Logit Accuracy: {}, SVM Accuracy: {}'.format(acc_logit, acc_svm))
    
    correct_logit /= len(target_data['train'])
    correct_svm /= len(target_data['train'])
    return correct_logit, correct_svm
#
        
#def test(epoch)
def test(epoch):
#    model.eval()
    test_loss = 0
    correct_logit = 0
    correct_svm = 0
    for batch_idx, data in enumerate(input_data['test']):
#        if args.cuda:
#            data, target = data.cuda(), target.cuda()
        target = target_data['test'][batch_idx]
        acc_logit = logit_model.score(data, target)
        acc_svm = svm_model.score(data, target)
        correct_logit += acc_logit
        correct_svm += acc_svm

    correct_logit /= len(target_data['test'])
    correct_svm /= len(target_data['test'])
    test_loss = test_loss
    test_loss /= len(target_data['test']) # loss function already averages over batch size
    print('Test: Logit Accuracy: {}, SVM Accuracy: {}'.format(correct_logit, correct_svm))
    return correct_logit, correct_svm

test_epochs = [1,5,10,20,30,50,70]

num_epochs = 50

train_props = np.linspace(0.01,1,20)

num_train_batches = len(all_input_data['train'])
train_inds = np.round(train_props*num_train_batches).astype(int)
num_train_inds = len(train_inds)

regularizer_vals= [1E-4, .001, .01, .1]
num_regularizer_vals = len(regularizer_vals)

error_analysis ={}
error_analysis['train_size'] = {}
error_analysis['reg_val'] = {}
error_analysis['train_size']['train'] = np.empty((num_epochs,num_train_inds,2))*np.nan
error_analysis['train_size']['test'] = np.empty((num_epochs,num_train_inds,2))*np.nan
error_analysis['reg_val']['train'] = np.empty((num_epochs,num_regularizer_vals,2))*np.nan
error_analysis['reg_val']['test'] = np.empty((num_epochs,num_regularizer_vals,2))*np.nan


#error_analysis = np.load('error_analysis.npy').item()

label1 = 'Logit'
label2 = 'SVM'
#label3 = 'Momentum .9'
 
lwidth = 3
axis_fontsize = 40
title_fontsize = 50



rc('xtick', labelsize=25) 
rc('ytick', labelsize=25) 

pyplot.figure(num=None, figsize=(10, 8), dpi=80)

for train_size_idx, train_size in enumerate(train_inds):
    
    input_data = {}
    target_data = {}
    input_data['train'] = all_input_data['train'][0:train_inds[train_size_idx]]
    target_data['train'] = all_target_data['train'][0:train_inds[train_size_idx]]
    input_data['test'] = all_input_data['test']
    target_data['test'] = all_target_data['test']
    
    logit_model = linear_model.SGDClassifier(loss = 'log', alpha=1E-4, warm_start=True)
    svm_model = linear_model.SGDClassifier(loss='hinge', alpha=1E-4, warm_start=True)
    
    for epoch in range(0,num_epochs):
        
        correct_logit, correct_svm = train(epoch)
        
        error_analysis['train_size']['train'][epoch, train_size_idx,0] = correct_logit
        error_analysis['train_size']['train'][epoch, train_size_idx,1] = correct_svm

        correct_logit, correct_svm = test(epoch)
        error_analysis['train_size']['test'][epoch, train_size_idx,0] = correct_logit
        error_analysis['train_size']['test'][epoch, train_size_idx,1] = correct_svm
        
        np.save('error_analysis', error_analysis )

#pyplot.subplot(121)
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

for reg_val_idx, reg_val in enumerate(regularizer_vals):
    
    input_data = {}
    target_data = {}
    input_data['train'] = all_input_data['train']
    target_data['train'] = all_target_data['train']
    input_data['test'] = all_input_data['test']
    target_data['test'] = all_target_data['test']
    
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
    
    epoch_range = range(0,num_epochs)
#    pyplot.subplot(num_regularizer_vals,1,reg_val_idx+1)
    pyplot.figure(num=None, figsize=(10, 8), dpi=80)
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
#pyplot.show()
#for epoch in range(1, 2):
#    epoch = 1
    
    
