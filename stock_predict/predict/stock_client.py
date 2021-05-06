#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  5 13:32:25 2021

@author: junkaiong
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  4 12:21:42 2021
@author: junkaiong
"""

import sys, os, requests, time, math, time
import datetime as dt
import yfinance as yf
import numpy as np
import pandas as pd
#import multiprocessing
#import threading
#from functools import partial
#import re

import socket
import sys
import requests
#import requests_oauthlib
import json


def send_stock_to_spark(http_resp, tcp_connection):
    for key, values in http_resp.items():
        try:
            stock_text = str(key) + '>' + str(values)
            print(stock_text)
            # print("Stock Text: " + stock_text)
            print ("------------------------------------------")
            tcp_connection.send(str(stock_text + '\n').encode())
            print(str(stock_text + '\n').encode())
            
        except:
            e = sys.exc_info()[0]
            print("Error: %s" % e)
            
            
# def send_stock_to_spark(http_resp, tcp_connection):
#     for line in http_resp.iter_lines():
#         try:
#             full_stock = json.loads(line)
#             stock_text = full_stock['text']
#             print("Stock Text: " + stock_text)
#             print ("------------------------------------------")
#             tcp_connection.send(stock_text + '\n')
#         except:
#             e = sys.exc_info()[0]
#             print("Error: %s" % e)

def get_stocks(tickers, start_date):
    x = yf.download(tickers, interval='1m', start=start_date, progress=False, group_by='ticker')
    # print(x)
    
    x_columns = x.columns.levels[0].to_list()
    # print(x_columns)
    
    ticker_dict = {}
    
    for i in range(len(x_columns)):
        new_x = x.iloc[-3,:][x_columns[i]][['Close', 'Volume']].to_dict()
        # print(new_x)
    
        ticker_dict[x_columns[i]] = new_x
    
    # print(ticker_dict)
    
    # still thinking if i should return as json string or as dict 
    # ticker_json_object = json.dumps(ticker_dict, indent = 4)  
    
    return ticker_dict

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

## this is to test if the dictionary is constructed correctly, without the TCP
# tickers = read_tickers('all')
# # tickers = 'AAPL GOOG'
# # start_date = '2021-05-03'
# start_date = str(dt.date.today())
# resp = get_stocks(tickers, start_date)

TCP_IP = "localhost"
TCP_PORT = 9009
conn = None
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((TCP_IP, TCP_PORT))
s.listen(1)
print("Waiting for TCP connection...")
conn, addr = s.accept()
print("Connected... Starting getting stocks.")

# tickers = read_tickers('all')
tickers = 'AAPL AMD'
#start_date = '2021-05-04'
start_date = str(dt.date.today()) # - dt.timedelta(days=1))
while True:
    resp = get_stocks(tickers, start_date)
    send_stock_to_spark(resp, conn)
    time.sleep(60)
