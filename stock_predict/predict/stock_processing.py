#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  4 15:56:56 2021

@author: junkaiong
"""

from pyspark import SparkConf,SparkContext
from pyspark.streaming import StreamingContext
from pyspark.sql import Row,SQLContext

import sys, os, requests, time, math
import datetime as dt
import yfinance as yf
import numpy as np
import pandas as pd

import tensorflow as tf
#from tensorflow.keras.layers import Dense, Flatten, Conv2D, LSTM
#from tensorflow.keras.models import Sequential, Model
#from tensorflow.keras.optimizers import Adam, SGD

## tensorflow WARNING, and ERROR messages are not printed
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# create spark configuration
conf = SparkConf()
conf.setAppName("StockStreamApp")
# create spark instance with the above configuration
sc = SparkContext(conf=conf)
sc.setLogLevel("ERROR")
# creat the Streaming Context from the above spark context with window size n seconds
ssc = StreamingContext(sc, 30)
# read datastream from socket
dataStream = ssc.socketTextStream("localhost",9009)


def aggregate_tags_count(new_values, total_sum):
    return sum(new_values) + (total_sum or 0)

def get_sql_context_instance(spark_context):
    if ('sqlContextSingletonInstance' not in globals()):
        globals()['sqlContextSingletonInstance'] = SQLContext(spark_context)
    return globals()['sqlContextSingletonInstance']

def process_rdd(time, rdd):
    print("----------- %s -----------" % str(time))

    #try:
    # Get spark sql singleton context from the current context
    sql_context = get_sql_context_instance(rdd.context)
    print('Processing RDD and Predicting Stock Price')

    #x = rdd.collect()
    #print(x)
    
    x_dict = rdd.collectAsMap()
    #print(x_dict)
    
    # open text file of dict and store as python dict
    file = open("xs_dict.txt", "r")
    contents = file.read()
    past_data_seq = eval(contents)
    file.close()
    
    #print(type(past_data_seq))

    for key, values in x_dict.items():
        for main_key, main_values in past_data_seq.items():
            if key == main_key:
                #print(key)
                
                # convert the latest minute's update value from string to dict type
                x_dict_value_toDict = eval(x_dict[key])
                #print(type(x_dict_value_toDict))
                
                # repack as a list item to append to past_data_seq of the ticker
                new_dict_value_list = [x_dict_value_toDict['Close'], x_dict_value_toDict['Volume']]
                #print(new_dict_value_list)

                #print(main_key)
                #print(past_data_seq[main_key][0])
                
                # drop the oldest element of the ticker's past_data_seq
                new_one_ticker_past_data_seq = past_data_seq[main_key][0][1:]
                #print(len(new_one_ticker_past_data_seq))

                # append the newest minute update from spark rdd stream into the ticker's past_data_seq
                new_one_ticker_past_data_seq.append(new_dict_value_list)
                #print(len(new_one_ticker_past_data_seq))

                past_data_seq[main_key][0] = new_one_ticker_past_data_seq
                #print(len(past_data_seq[main_key][0]))

                #print(past_data_seq[main_key][0])
                
                # change the type of the single ticker sequence into a numpy array
                new_one_ticker_past_data_seq = np.array(new_one_ticker_past_data_seq)
                new_one_ticker_past_data_seq =np.reshape(new_one_ticker_past_data_seq, (1,60,2))
                #print(new_one_ticker_past_data_seq.shape)

                
                # #load in the saved model and predict price
                
                loaded_toy_model = tf.keras.models.load_model("toy_model.h5")
                # print(type(loaded_toy_model))
                # loaded_toy_model.summary()
                pred_price = loaded_toy_model.predict(new_one_ticker_past_data_seq)
                
                print("----------- %s -----------" % str(time))
                print('The predicted price of '+  key+ ' is '+ str(pred_price[0][0]))
                #print(pred_price[0][0])
                
    # save the python dict of xs as a txt file
    geeky_file = open('xs_dict.txt', 'wt')
    data = str(past_data_seq)
    geeky_file.write(data)
    geeky_file.close()




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

def get_prev_day_stocks(tickers, start_date, target_min=5, seq_len=60, feats=['Close', 'Volume']):
    data_all = yf.download(tickers, interval='1m', start=start_date, progress=False, group_by='ticker')
    # print(data_all)
    
    last_60min = data_all.iloc[-67:-1,:]
    # print(last_60min)
    
    # initialize xs as empty dictionary
    xs = {}

    for t in tickers:
        if len(tickers) > 1:
            data_sub = last_60min[t]
        else:
            data_sub = last_60min
        x_seq = generate_sequences(data_sub, target_min=target_min, seq_len=seq_len, feats=feats)
        
        x_seq = x_seq.tolist()
        
        xs[t] = x_seq
        
    # save the python dict of xs as a txt file
    geeky_file = open('xs_dict.txt', 'wt')
    data = str(xs)
    geeky_file.write(data)
    geeky_file.close()
    
    return xs


#tickers = read_tickers('all')
tickers = ['AAPL', 'AMD']
date_today = dt.date.today() # - dt.timedelta(days=1)
prev_day = date_today - dt.timedelta(days=1)
xs = get_prev_day_stocks(tickers, prev_day, target_min=5, seq_len=60, feats=['Close', 'Volume'])

#print(len(xs))

words = dataStream.map(lambda line: (line.encode("ascii", "ignore").split(">")[0], line.encode("ascii", "ignore").split(">")[1]))

# # print in the period 
print("datastream RDD received: ")
words.pprint(10)

# # do processing for each RDD generated in each interval
words.foreachRDD(process_rdd)

# start the streaming computation
ssc.start()
# wait for the streaming to finish
ssc.awaitTermination()

