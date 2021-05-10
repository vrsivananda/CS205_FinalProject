## referenced Harvard CS205 Course Material Lecture C3
## https://harvard-iacs.github.io/2021-CS205/lectures/C3/ 

import sys, os, requests, time, math, time
import datetime as dt
import yfinance as yf
import numpy as np
import pandas as pd


import socket
import sys
import requests

import json


# this method sends the latest stock data to spark
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
            
            

# this method gets the latest stock data from yfinance
def get_stocks(tickers, start_date):
    x = yf.download(tickers, interval='1m', start=start_date, progress=False, group_by='ticker')
    # drop the nan values from the read off yfinance data
    x.dropna(inplace = True)
    
    # get the columns of ticker names
    x_columns = x.columns.levels[0].to_list()

    ticker_dict = {}
    
    for i in range(len(x_columns)):
        # defensive programming to check that the data sequence is not empty
        if len(x)>0:
            new_x = x.iloc[-1,:][x_columns[i]][['Close', 'Volume']].to_dict()
        
        # if data sequence is empty, then set the fields to zeros
        else:
            new_x = {'Close': 0, 'Volume':0}
    
        ticker_dict[x_columns[i]] = new_x
    
    return ticker_dict

# this method reads in the names of tickers
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


TCP_IP = "localhost"
TCP_PORT = 9009
conn = None
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((TCP_IP, TCP_PORT))
s.listen(1)
print("Waiting for TCP connection...")
conn, addr = s.accept()
print("Connected... Starting getting stocks.")

tickers = read_tickers('all')
tickers = tickers[0:10]
#tickers = 'AAPL AMD GOOG'

start_date = str(dt.date.today()) # - dt.timedelta(days=1))
while True:
    resp = get_stocks(tickers, start_date)
    send_stock_to_spark(resp, conn)
    time.sleep(20)#60)
