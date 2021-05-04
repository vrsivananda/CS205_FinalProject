#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  4 15:56:56 2021

@author: junkaiong
"""

from pyspark import SparkConf,SparkContext
from pyspark.streaming import StreamingContext
from pyspark.sql import Row,SQLContext
import sys
import requests

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
    print('test2')
    print(rdd.collect())
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

