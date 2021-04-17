import sys, os, requests
import datetime as dt
import yfinance as yf
import numpy as np
#import time
# Author: Kevin Hare
# Last Updated: 4/16/2021
# Purpose: Save yfinance data and convert to sequence

def find_last_date():
    """Returns last date stock prices were read to train
    data and load most recent entries"""
    files = os.listdir()
    if 'last_date' not in files:
        today = dt.date.today()
        with open('last_date', 'a') as f:
            f.write(str(today))
            f.close()
        return today
    else:
        with open('last_date', 'r') as f:
            l = f.read().strip()
            return l

def read_tickers():
    """Reads ticker list"""
    url = 'https://pkgstore.datahub.io/core/s-and-p-500-companies/'+\
        'constituents_json/data/87cab5b5abab6c61eafa6dfdfa068a42/constituents_json.json'
    files = os.listdir()
    if 'ticker_list' not in files:
        t = requests.get(url).json()
        tickers = [d['Symbol'] for d in t]
        with open('ticker_list', 'w') as f:
            for ticker in tickers:
                f.write(ticker + '\n')
            f.close()
    else:
        tickers = []
        f = open('ticker_list', 'r')
        for line in f:
            tickers.append(line.strip())
    print(tickers)
    return tickers

def process_data(tickers, datapath, seq_len=60, save=True):
    """Read stock data, convert to sequence, and save out
    Inputs
    ------
    tickers: iterable, ticker symbols to iterate over, required"""
    today = dt.date.today()
    last_date = dt.datetime.strptime(find_last_date(), "%Y-%m-%d").date()
    for t in tickers:
        day = last_date
        while day < today:
            end_day = day + dt.timedelta(days=1)
            data = yf.download(t, period='1d', interval='1m', start=str(day), end=str(end_day))
            day += dt.timedelta(days=1)

            # Need to except weekends
            if len(data) == 0:
                continue

            # Generate sequences & save
            s = np.lib.stride_tricks.sliding_window_view(data, (seq_len, data.shape[1])).squeeze(axis=1)

            try:
                os.mkdir( datapath + 'raw_seq')
            except:
                pass

            np.save(datapath + 'raw_seq/' + t + '_' + str(day) + '.npy', s)
            
            







if __name__ == '__main__':
    tickers = ['MSFT']
    datapath = '../../data/'
    process_data(tickers, datapath)

