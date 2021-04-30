#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 00:38:16 2021

@author: junkaiong
"""

from data_creater import *


stocks = companies()
tickers = stocks.values.tolist()

#Select stock to perform tests
ticker = tickers[2][1]

print("Stock ticker selected for testing: {}".format(ticker))


# this section is to construct toy models for testing of the sequences
## ********************************************************************
import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Bidirectional
from keras.optimizers import RMSprop

def fixed_model(X,y, learn_rate):
    model = Sequential()
    model.add(LSTM(5,input_shape=(X.shape[1:])))
    model.add(Dense(y.shape[1], activation='tanh'))
      
    # compile the model
    optimizer = RMSprop(lr=learn_rate)
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model

def dynamic_model(X,y, learn_rate):
    model = Sequential()
    model.add(LSTM(X.shape[1],input_shape=(X.shape[1:])))
    model.add(Dense(y.shape[1], activation='tanh'))
      
    # compile the model
    optimizer = RMSprop(lr=learn_rate)
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model

def bidirectional_model(X,y, learn_rate):
    model = Sequential()
    model.add(Bidirectional(LSTM(X.shape[1],return_sequences=False), input_shape=(X.shape[1:])))
    model.add(Dense(X.shape[1]))
    model.add(Dense(y.shape[1], activation='tanh'))
      
    # compile the model
    optimizer = RMSprop(lr=learn_rate)
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model

def stacked_model(X,y, learn_rate):
    model = Sequential()
    model.add(LSTM(10,return_sequences=True, input_shape=(X.shape[1:])))
    model.add(LSTM(5))
    model.add(Dense(y.shape[1], activation='tanh'))
      
    # compile the model
    optimizer = RMSprop(lr=learn_rate)
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model

#Create list of our models for use by the testing function.
models =[]
models.append(("Fixed",fixed_model))
models.append(("Dynamic",dynamic_model))
models.append(("Bidirectional",bidirectional_model))
models.append(("Stacked",stacked_model))

## ********************************************************************




# this section is to create functions to test the model prediction using the sequences created
## ********************************************************************
from collections import OrderedDict

def test_model(ticker,epochs,models,seq,window_sizes):
    #test result data
    sizes = []
    #seq_name = []
    model_name = []
    train_errors = []
    test_errors = []
    param_count = []
    
    for window_size in window_sizes:
        print("\nWindow size: {}".format(window_size))
        print('----------------')
        for model_item in models:
            seq_obj = seq[1](ticker,window_size,1)
            X_train,y_train,X_test,y_test = split_data(seq_obj)
            model = model_item[1](X_train,y_train,0.001)
            
            # fit model!
            model.fit(X_train, y_train, epochs=epochs, batch_size=50, verbose=0)

            # print out training and testing errors
            training_error = model.evaluate(X_train, y_train, verbose=0)
            testing_error = model.evaluate(X_test, y_test, verbose=0)
            msg = " > Model: {0:<15} Param count: {1:} \tTraining error: {2:.4f}\tTesting error: {3:.4f}"
            print(msg.format(model_item[0],model.count_params(),training_error,testing_error))

            #update result variables
            param_count.append(model.count_params())
            sizes.append(window_size)
            #seq_name.append(seq[0])
            model_name.append(model_item[0])
            train_errors.append(float("{0:.4f}".format(training_error)))
            test_errors.append(float("{0:.4f}".format( testing_error)))

    table= OrderedDict()
    table['Window Size'] = sizes
    table['Sequence Name'] =  [seq[0] for _ in range(len(sizes))]
    table['Model Name'] = model_name
    table['Ticker'] = [ticker for _ in range(len(sizes))]
    table['Training Error'] = train_errors
    table['Testing Error'] = test_errors
    table['Param Count'] = param_count
        
    return table


def update_test_table(*argv):
    file_path = "./data/model_test.csv"
    
    table = pd.read_csv(file_path)
    tickers = set( table['Ticker'].values.tolist())
    
    for item in argv:

        #first check if already exist 
        check = item['Ticker'][0]
        if check in tickers:
            #drop items
            idx = table[(table['Ticker']== check)  &  (table['Sequence Name']== item['Sequence Name'][0])].index
            table =  table.drop(idx)

        #append current test
        table = table.append(pd.DataFrame(item))

    table = table.reset_index(drop=True)
    table.to_csv(file_path, index = False)

def get_test_table():
    file_path = "./data/model_test.csv"
    return pd.read_csv(file_path)
## ********************************************************************



# this section creates a Sequence using the class "SimpleSequence" from data_creater.py
# test the model using the sequence created
## ********************************************************************
seed = 7
np.random.seed(seed)

#Model testing variables
epochs =100
window_sizes =[5,7,10,20]


print("*** Simple Sequence Model Test for {} ***".format(ticker))
print("=" * 45)

seq_name = ('Simple',SimpleSequence)

test_1  = test_model(ticker,epochs,models,seq_name,window_sizes)
update_test_table(test_1)
## ********************************************************************


# this section creates a Sequence using the class "MultiSequence" from data_creater.py
# test the model using the sequence created
## ********************************************************************
print("*** Multi Sequence Model Test for {} ***".format(ticker))
print("=" * 45)

seq_name = ('Multi',MultiSequence)

test_2  = test_model(ticker,epochs,models,seq_name,window_sizes)
update_test_table(test_2)
## ********************************************************************
