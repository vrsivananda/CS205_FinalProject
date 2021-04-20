import sys, os, requests, time
import datetime as dt
import yfinance as yf
import numpy as np
import multiprocessing
from functools import partial
# Author: Kevin Hare
# Last Updated: 4/17/2021
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

def read_tickers(which=None):
    """Reads ticker list"""
    if which != "all":
        return which.split()
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

    return tickers

def process_tickers(tickers, datapath, seq_len=60, target_min=5, save=True):
    """Read stock data, convert to sequence, and save out
    Inputs
    ------
    tickers: iterable, ticker symbols to iterate over, required
    datapath: path to data storage files
    seq_len: (optional) length of sequence in minutes
    target_min: (optional) target minutes ahead (default set at 5 min)"""
    t1 = time.time()
    today = dt.date.today()
    last_date = dt.datetime.strptime(find_last_date(), "%Y-%m-%d").date()
    for t in tickers:
        day = last_date
        xs, ys = [], []
        while day < today:
            end_day = day + dt.timedelta(days=1)
            data = yf.download(t, period='1d', interval='1m', start=str(day), end=str(end_day), progress=False)
            day += dt.timedelta(days=1)

            # Need to except weekends
            if len(data) == 0:
                continue

            # Generate sequences & save
            #s = np.lib.stride_tricks.sliding_window_view(data, (seq_len, data.shape[1])).squeeze(axis=1)
            for i, v in enumerate(range(target_min, len(data)-seq_len)):
                if i == 0:
                    x = np.expand_dims(data.values[i:i+seq_len, :], axis=0)
                else:
                    z = np.expand_dims(data.values[i:i+seq_len, :], axis=0)
                    x = np.concatenate((x, z), axis=0)
            
            y = data['Close'].values[seq_len + target_min:]

            xs.append(x)
            ys.append(y)
        xs = np.concatenate(xs, axis=0)
        ys = np.concatenate(ys, axis=0)
        try:
            os.mkdir( datapath + 'raw_seq')
        except:
            pass

        np.savez(datapath + 'raw_seq/' + t + '.npz', x=xs, y=ys)
    # Return processing time
    return time.time()-t1

def process_ticker(t, datapath, seq_len=60, target_min=5, save=True):
    """Processes stock price data for a single ticker
    Inputs
    ------
    t: ticker symbols to iterate over, required
    datapath: path to data storage files
    seq_len: (optional) length of sequence in minutes
    target_min: (optional) target minutes ahead (default set at 5 min)"""
    today = dt.date.today()
    last_date = dt.datetime.strptime(find_last_date(), "%Y-%m-%d").date()
    day = last_date
    day_ct = 0
    # Create list of xs and ys to be combined into array
    xs, ys = [], []
    while day < today:
        end_day = day + dt.timedelta(days=1)
        data = yf.download(t, period='1d', interval='1m', start=str(day), end=str(end_day), progress=False)
        day += dt.timedelta(days=1)

        # Need to except weekends
        if len(data) == 0:
            continue

        # Generate sequences & save
        #s = np.lib.stride_tricks.sliding_window_view(data, (seq_len, data.shape[1])).squeeze(axis=1)
        for i, v in enumerate(range(target_min, len(data)-seq_len)):
            if i == 0:
                x = np.expand_dims(data.values[i:i+seq_len, :], axis=0)
            else:
                z = np.expand_dims(data.values[i:i+seq_len, :], axis=0)
                x = np.concatenate((x, z), axis=0)
        
        y = data['Close'].values[seq_len + target_min:]
        xs.append(x)
        ys.append(y)
    xs = np.concatenate(xs, axis=0)
    ys = np.concatenate(ys, axis=0)
    try:
        os.mkdir( datapath + 'raw_seq')
    except:
        pass
    np.savez(datapath + 'raw_seq/' + t + '.npz', x=xs, y=ys)

def process_data_multi(tickers, num_proc=None, datapath='./', seq_len=60, target_min=5, save=True):
    """Applies multiprocessing to tickers fed into a ticker list, wraps the process ticker, 
    uses functools.partial() to apply closure pattern to single ticker processing

    Note: Used functools.partial() rather than self-closure for optimization purposes
    
    Returns
    -------
    proc_time: time in seconds to perform processing"""
    t1 = time.time()
    p = multiprocessing.Pool(num_proc)
    mapfunc = partial(process_ticker, datapath=datapath, seq_len=seq_len, target_min=target_min, save=True)
    p.map(mapfunc, tickers)

    proc_time = (time.time()-t1)
    return proc_time


def combine_seqs(datapath):
    """Combines sequences"""
    # NON SPARK! WORK IN PROGRESS
    files = os.listdir(datapath + 'raw_seq/')
    for i, fn in enumerate(files):
        print(i)
        if i == 0:
            npfiles = np.load(datapath + 'raw_seq/' + fn, allow_pickle=True)
            x_data, y_data = npfiles['x'], npfiles['y']
        else:
            d_add = np.load(datapath + 'raw_seq/' + fn, allow_pickle=True)
            x_add, y_add = d_add['x'], d_add['y']
            x_data = np.concatenate((x_data, x_add), axis=0)
            y_data = np.concatenate((y_data, y_add), axis=0)
        if i == 20:
            break
    np.savez(datapath + 'training_data.npz', x_train=x_data, y_train=y_data)#, allow_pickle=True)      
            
if __name__ == '__main__':
    t1 = time.time()
    tickers = read_tickers('all')
    datapath = './'
    #process_tickers(tickers[:20], datapath)
    proctime = process_data_multi(tickers[:20], num_proc=8, datapath='./', seq_len=60, target_min=5, save=True)
    #print(proctime)
    combine_seqs(datapath)
    print(time.time()-t1)