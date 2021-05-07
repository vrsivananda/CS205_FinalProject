#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  3 17:39:01 2021

@author: junkaiong
"""

    
import sys, os, requests, time, math
import datetime as dt
import yfinance as yf
import numpy as np
import pandas as pd
import multiprocessing
import threading
from functools import partial
import re

from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Bidirectional
from keras.optimizers import RMSprop


#from pyspark import SparkConf, SparkContext
#from pyspark.sql import SQLContext, SparkSession
#from pyspark.sql.functions import split as ps_split
#from pyspark.sql.functions import when
#from pyspark.sql.functions import collect_list

# Author: Kevin Hare
# Last Updated: 5/1/2021
# Purpose: Save yfinance data and convert to sequence

def find_last_date(interval='1m'):
    """Returns date 30 days prior, as this will be the only available
    date for pulling data
    Args:
        interval: interval to pull data at; will correspond 
                  mechanically to the last day available
    Returns:
        ld: date in string form"""
    int2day = {'1m': 29, '2m': 60}
    ld = dt.date.today() - dt.timedelta(days=int2day[interval])
    # Read out day for reference
    with open('last_date', 'w') as f:
        f.write(str(ld))
        f.close()
    return str(ld)

def read_tickers(which=None):
    """Reads ticker list"""
    if which != "all":
        return which.split()
    url = 'https://pkgstore.datahub.io/core/s-and-p-500-companies/'+\
        'constituents_json/data/87cab5b5abab6c61eafa6dfdfa068a42/constituents_json.json'
    files = os.listdir()
    if 'ticker_list' not in files:
        t = requests.get(url).json()
        tickers = [d['Symbol'] for d in t if '.' not in d['Symbol']]
        with open('ticker_list', 'w') as f:
            for ticker in tickers:
                f.write(ticker + '\n')
            f.close()
    else:
        tickers = []
        f = open('ticker_list', 'r')
        for line in f:
            tickers.append(line.strip())

    return tickers

def generate_sequences(data, target_min=5, seq_len=60, feats=['Close', 'Volume']):
    """Given a subset of data for a particular ticker and date, 
    will return sequences of the appropriate length
    
    Args
        data: subset dataframe
        target_min (optional): target minutes ahead of end of sequence
        seq_len (optional): sequence length to consider
        feats (optional): list of features to keep
    Returns
        seqs: numpy array of sequences (n_seqs, seq_len, len(feats))
    """
    for i, v in enumerate(range(target_min, len(data)-seq_len)):
        if i == 0:
            x = np.expand_dims(data[feats].values[i:i+seq_len, :], axis=0)
        else:
            z = np.expand_dims(data[feats].values[i:i+seq_len, :], axis=0)
            x = np.concatenate((x, z), axis=0)
    return x


def process_tickers(tickers, start_date, end_date, seq_len=60, target_min=5, feats=['Close', 'Volume'], save=True):
    """Processes a list of tickers between start and end date. Will return a set of sequences and targets of size
    `(n_ticker*(end_date-start_date)*n_seqs, seq_len, n_feats)`. Makes the assumption that tickers do not need
    to be linked between days
    
    Args:
        tickers (list): list of ticker symbols to be processed
        start_date (str): YYYY-MM-DD string formatted date for start of pull
        end_date (str): YYYY-MM-DD string formattted date for end of pull
        seq_len (optional, int): number of periods to include in each sequence, defaults to 60 (i.e. one hour)
        target_min (optional, int): minutes ahead of end of sequence to predict
        feats (optional, list): features to keep for the model
    
    Returns:
        (X, y) where the elements are as follows:
            X (np.array): array concatenated by date and ticker (3-Dim)
            y (np.array): array concatenated by date and ticker (1-Dim)
    """
    # Download data for tickers
    data_all = yf.download(tickers, period='1d', interval='1m', start=str(start_date), 
                            end=str(end_date), progress=False, group_by='ticker')
    
    # Iterate through dataframe
    xs, ys = [], []
    data_all.index = pd.to_datetime(data_all.index)
    dates = np.unique(data_all.index.date)
    for t in tickers:
        for d in dates:
            if len(tickers) > 1:
                data_sub = data_all[t].loc[str(d)]
            else:
                data_sub = data_all.loc[str(d)]
            x = generate_sequences(data_sub, target_min=5, seq_len=60, feats=['Close', 'Volume'])
            y = data_sub['Close'].values[seq_len + target_min:]
            
            # Add to existing sequences
            xs.append(x)
            ys.append(y)
    xs = np.concatenate(xs, axis=0)
    ys = np.concatenate(ys, axis=0)
    # xs = np.array(xs)
    # ys = np.array(ys)
    return xs, ys

# def process_data_seq(tickers, seq_len=60, target_min=5, feats=['Close', 'Volume'], save=True, datapath='./'):
#     """Sequential version of data pull and save function. This version pulls all N tickers
#     for each date in the applicable date sequences, converts to sequences, and saves. yfinance
#     API parallelizes threads to improve performance, but processing still slowed by Python limits
#     Args:
#         tickers (list): List of tickers for data downloads
#         seq_len (optional, int): length in minutes of sequences for training
#         target_min (optional, int): minutes ahead of end of training seqeuence for target price
#         feats (optional, list[str]): list of features to include from yfinance
#         save (optional, boolean): will save out file if True, otherwise will not
#         datapath (optional, str): path to save training_data.npz; defaults to current directory.
#     Returns
#         None.
#     """
#     t1 = time.time()
#     last_date = dt.datetime.strptime(find_last_date(), "%Y-%m-%d").date()

#     # Set variable for today's date
#     # Set beginning and end start dates
#     start_date = last_date
#     end_date = start_date + dt.timedelta(days=7)
#     today_dt = dt.datetime.today().date()

#     total_x, total_y = [], []
#     # Loop over date ranges
#     while start_date < today_dt:
#         print(start_date, end_date)
#         x, y = process_tickers(tickers, start_date, end_date, seq_len=60, target_min=5, feats=['Close', 'Volume'])

#         # Increment dates - note additional day on start date to make non-overlapping days
#         start_date = end_date + dt.timedelta(days=1)
#         end_date += dt.timedelta(days=8)

#         total_x.append(x)
#         total_y.append(y)
#     # Convert to single array and save out
#     total_x = np.concatenate(total_x, axis=0)
#     total_y = np.concatenate(total_y, axis=0)
#     if save == True:
#         np.savez_compressed(datapath + 'training_data.npz', x_train=total_x, y_train=total_y)
    

# def process_data_parallel(tickers, n_proc=1, seq_len=60, target_min=5, feats=['Close', 'Volume'], save=True, datapath='./'):
#     """Parallel version of data pull and save function. Maps n_proc processes to roughly evenly divided
#     blocks of the tickers. Note that yfinance API will automatically pull as single DF (i.e. non Multi-Index)
#     if only one ticker, so minimum number must be two. 
#     Args:
#         tickers (list): List of tickers for data downloads
#         n_proc (optional, int): number of processes to map. Greater than the number of CPU cores
#                                 will result in no additional speedup
#         seq_len (optional, int): length in minutes of sequences for training
#         target_min (optional, int): minutes ahead of end of training seqeuence for target price
#         feats (optional, list[str]): list of features to include from yfinance
#         save (optional, boolean): will save out file if True, otherwise will not
#         datapath (optional, str): path to save training_data.npz; defaults to current directory.
#     Returns
#         None.
#     """
#     t1 = time.time()
#     first_date = dt.datetime.strptime(find_last_date(), "%Y-%m-%d").date()

#     # Chunk list of tickers into relatively evenly spaced sizes
#     # Note: If there are more processors specified than tickers,
#     # make adjustment to require that each processor pull at least
#     # two tickers
#     if n_proc >= len(tickers):
#             size = 2
#             n_proc = int(len(tickers)/size)
#     else:
#         size = math.ceil(len(tickers)/n_proc)
#     zs = [tickers[i:i+size] for i in range(0, len(tickers), size)]
    
#     # Set variable for today's date
#     # Set beginning and end start dates
#     start_date = first_date
#     end_date = start_date + dt.timedelta(days=7)
#     today_dt = dt.datetime.today().date()
    
#     total_x, total_y = [], []
#     while start_date < today_dt:
#         print(start_date, end_date)
        
#         # Use functools.partial method to create mapping function
#         # for non-simple functions
#         # See more here: https://docs.python.org/3/library/functools.html
#         mapfunc = partial(process_tickers, start_date=start_date, end_date=end_date, seq_len=seq_len,
#                                 target_min=target_min, feats=['Close', 'Volume'], save=True)
#         with multiprocessing.Pool(n_proc) as p:
#             vals = p.map(mapfunc, zs)
#             p.close()
#             p.join()

#         # Concatenate results together into single array
#         xs = np.concatenate([v[0] for v in vals], axis=0)
#         ys = np.concatenate([v[1] for v in vals], axis=0)

#         start_date = end_date + dt.timedelta(days=1)
#         end_date += dt.timedelta(days=8)

#         total_x.append(xs)
#         total_y.append(ys)

#     # Convert to single array and save out
#     total_x = np.concatenate(total_x, axis=0)
#     total_y = np.concatenate(total_y, axis=0)

#     if save == True:
#         np.savez_compressed(datapath + 'training_data.npz', x_train=total_x, y_train=total_y)    
        
        
def process_data_seq_2day(tickers, seq_len=60, target_min=5, feats=['Close', 'Volume'], save=True, datapath='./'):
    """Sequential version of data pull and save function. This version pulls all N tickers
    for each date in the applicable date sequences, converts to sequences, and saves. yfinance
    API parallelizes threads to improve performance, but processing still slowed by Python limits
    Args:
        tickers (list): List of tickers for data downloads
        seq_len (optional, int): length in minutes of sequences for training
        target_min (optional, int): minutes ahead of end of training seqeuence for target price
        feats (optional, list[str]): list of features to include from yfinance
        save (optional, boolean): will save out file if True, otherwise will not
        datapath (optional, str): path to save training_data.npz; defaults to current directory.
    Returns
        None.
    """
    t1 = time.time()
    last_date = dt.datetime.strptime(find_last_date(), "%Y-%m-%d").date()

    # Set variable for today's date
    # Set beginning and end start dates
    today_dt = dt.datetime.today().date()
    start_date = today_dt - dt.timedelta(days=2)
    end_date = today_dt
    
    # start_date = last_date
    # end_date = start_date + dt.timedelta(days=7)
    # today_dt = dt.datetime.today().date()

    total_x, total_y = [], []
    # Loop over date ranges
    while start_date < today_dt:
        print(start_date, end_date)
        x, y = process_tickers(tickers, start_date, end_date, seq_len=60, target_min=5, feats=['Close', 'Volume'])

        # Increment dates - note additional day on start date to make non-overlapping days
        start_date = end_date + dt.timedelta(days=1)
        end_date += dt.timedelta(days=8)

        total_x.append(x)
        total_y.append(y)
    # Convert to single array and save out
    total_x = np.concatenate(total_x, axis=0)
    total_y = np.concatenate(total_y, axis=0)
    
    print(total_x[1].shape)
    print(total_y.shape)
    
    return total_x, total_y
    # if save == True:
    #     np.savez_compressed(datapath + 'training_data.npz', x_train=total_x, y_train=total_y)



# this section is to construct toy models for testing of the sequences
## ********************************************************************

def fixed_model(X,y, learn_rate):
    model = Sequential()
    model.add(LSTM(5,input_shape=X[1].shape))
    # model.add(LSTM(5,input_shape=(X.shape[1:])))
    model.add(Dense(1, activation='tanh'))
      
    # compile the model
    optimizer = RMSprop(lr=learn_rate)
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model

# def dynamic_model(X,y, learn_rate):
#     model = Sequential()
#     model.add(LSTM(X.shape[1],input_shape=(X.shape[1:])))
#     model.add(Dense(1, activation='tanh'))
      
#     # compile the model
#     optimizer = RMSprop(lr=learn_rate)
#     model.compile(loss='mean_squared_error', optimizer=optimizer)
#     return model

# def bidirectional_model(X,y, learn_rate):
#     model = Sequential()
#     model.add(Bidirectional(LSTM(X.shape[1],return_sequences=False), input_shape=(X.shape[1:])))
#     model.add(Dense(X.shape[1]))
#     model.add(Dense(1, activation='tanh'))
      
#     # compile the model
#     optimizer = RMSprop(lr=learn_rate)
#     model.compile(loss='mean_squared_error', optimizer=optimizer)
#     return model

# def stacked_model(X,y, learn_rate):
#     model = Sequential()
#     model.add(LSTM(10,return_sequences=True, input_shape=(X.shape[1:])))
#     model.add(LSTM(5))
#     model.add(Dense(1, activation='tanh'))
      
#     # compile the model
#     optimizer = RMSprop(lr=learn_rate)
#     model.compile(loss='mean_squared_error', optimizer=optimizer)
#     return model

#Create list of our models for use by the testing function.
models =[]
models.append(("Fixed",fixed_model))
# models.append(("Dynamic",dynamic_model))
# models.append(("Bidirectional",bidirectional_model))
# models.append(("Stacked",stacked_model))

## ********************************************************************

# this section is to create functions to test the model prediction using the sequences created
## ********************************************************************
from collections import OrderedDict

def test_model(ticker,epochs,models,seq_x, seq_y, window_sizes):
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
            # seq_obj = seq[1](ticker,window_size,1)
            # X_train,y_train,X_test,y_test = split_data(seq_obj)
            
            model = model_item[1](seq_x, seq_y,0.001)
            
            # fit model!
            model.fit(seq_x, seq_y, epochs=epochs, batch_size=50, verbose=0)

            # print out training and testing errors
            training_error = model.evaluate(seq_x, seq_y, verbose=0)
            testing_error = model.evaluate(seq_x, seq_y, verbose=0)
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
    table['Model Name'] = model_name
    table['Ticker'] = [ticker for _ in range(len(sizes))]
    table['Training Error'] = train_errors
    table['Testing Error'] = test_errors
    table['Param Count'] = param_count
        
    return table

## ********************************************************************
# parameters to test the model using the sequence created
## ********************************************************************
seed = 7
np.random.seed(seed)

#Model testing variables
epochs =100
window_sizes =[5,7,10,20]



## ********************************************************************        
            
if __name__ == '__main__':
    t1 = time.time()
    # tickers = read_tickers('all')
    tickers = ['AAPL']
    datapath = './'
    try:
        # if sys.argv[1] != 'all':
        #     tickers = tickers[:int(sys.argv[1])]
        seq_x, seq_y = process_data_seq_2day(tickers)
        test_1  = test_model(tickers,epochs,models,seq_x, seq_y,window_sizes)


    except IndexError:
        print("index error")


    print(f'Total time: {time.time()-t1:0.2f}')
