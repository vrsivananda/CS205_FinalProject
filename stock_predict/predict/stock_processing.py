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

# create spark configuration
conf = SparkConf()
conf.setAppName("StockStreamApp")
# create spark instance with the above configuration
sc = SparkContext(conf=conf)
sc.setLogLevel("ERROR")
# creat the Streaming Context from the above spark context with window size n seconds
ssc = StreamingContext(sc, 30)
# setting a checkpoint to allow RDD recovery
#ssc.checkpoint("checkpoint_StockApp")
# read data from port 9009
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
    print('test1')
    # convert the RDD to Row RDD
    #row_rdd = rdd.map(lambda w: Row(hashtag=w[0], hashtag_count=w[1]))
    #nrdd = rdd.map(lambda x: x.decode())
    # print('test2')
    x = rdd.collect()
    print(x)
    # create a DF from the Row RDD
    #hashtags_df = sql_context.createDataFrame(row_rdd)
    
    # Register the dataframe as table
    #hashtags_df.registerTempTable("hashtags")
    # get the top 10 hashtags from the table using SQL and print them
    #hashtag_counts_df = sql_context.sql("select hashtag, hashtag_count from hashtags order by hashtag_count desc limit 10")
    #hashtag_counts_df.show()
    #except:
    #    e = sys.exc_info()[0]
    #    print("Error: %s" % e)


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

    # Iterate through dataframe
    xs, ys = [], []

    for t in tickers:
        if len(tickers) > 1:
            data_sub = last_60min[t]
        else:
            data_sub = last_60min
        x_seq = generate_sequences(data_sub, target_min=target_min, seq_len=seq_len, feats=feats)
        y = data_sub['Close'].values[seq_len + target_min:]
        
        # Add to existing sequences
        xs.append(x_seq)
        ys.append(y)
    xs = np.concatenate(xs, axis=0)
    ys = np.concatenate(ys, axis=0)
    

    print(xs.shape)
    return xs, ys


# tickers = read_tickers('all')
tickers = ['AAPL']
prev_day = dt.date.today() - dt.timedelta(days=1)
xs, ys = get_prev_day_stocks(tickers, prev_day, target_min=5, seq_len=60, feats=['Close', 'Volume'])

print(xs)

# # to uncomment below to stream
# split each tweet into words
words = dataStream.map(lambda x: x.split('\n'))#.flatMap(lambda line: line.split(">"))

# # filter the words to get only hashtags, then map each hashtag to be a pair of (hashtag,1)
#ashtags = words.filter(lambda w: '#' in w).map(lambda x: (x, 1))
# # add in the period 
#hashtags = hashtags.reduceByKey(lambda x, y: x + y)
# # print in the period 
#hashtags.pprint(100)
words.pprint(10)

# # acculmulate the state
#tags_totals = hashtags.updateStateByKey(aggregate_tags_count)
# # do processing for each RDD generated in each interval
#tags_totals.foreachRDD(process_rdd)
words.foreachRDD(process_rdd)

# start the streaming computation
ssc.start()
# wait for the streaming to finish
ssc.awaitTermination()

