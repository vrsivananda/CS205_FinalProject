import sys, os, requests, time
import datetime as dt
import yfinance as yf
import numpy as np
import multiprocessing
from functools import partial
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext, SparkSession
from pyspark.sql.functions import split as ps_split
from pyspark.sql.functions import when
from pyspark.sql.functions import collect_list
import re
# Author: Kevin Hare
# Last Updated: 4/29/2021
# Purpose: Save yfinance data and convert to sequence

def find_last_date(interval='1m'):
    """Returns date 30 days prior, as this will be the only available
    date for pulling data
    Args:
        interval: interval to pull data at; will correspond 
                  mechanically to the last day available
    Returns:
        ld: date in string form"""
    int2day = {'1m': 30, '2m': 60}
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

def process_tickers(tickers, datapath, seq_len=60, target_min=5, feats=['Close', 'Volume'], save=True):
    """Read stock data, convert to sequence, and save out
    Args
        tickers: iterable, ticker symbols to iterate over, required
        datapath: path to data storage files
        seq_len: (optional) length of sequence in minutes
        target_min: (optional) target minutes ahead (default set at 5 min)
        feats: (optional) features to keep from sequence. Due to high resolution nature
                of the problem, this defaults to only close & volume
    Returns:
        processing time, in seconds
    """
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
                    x = np.expand_dims(data[feats].values[i:i+seq_len, :], axis=0)
                else:
                    z = np.expand_dims(data[feats].values[i:i+seq_len, :], axis=0)
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

def process_ticker(t, datapath, last_date, seq_len=60, target_min=5, feats=['Close', 'Volume'], save=True):
    """Processes stock price data for a single ticker
    Args
        t: ticker symbols to iterate over, required
        datapath: path to data storage files
        seq_len: (optional) length of sequence in minutes
        target_min: (optional) target minutes ahead (default set at 5 min)
    Returns
        None
    """
    today = dt.date.today()
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
                x = np.expand_dims(data[feats].values[i:i+seq_len, :], axis=0)
            else:
                z = np.expand_dims(data[feats].values[i:i+seq_len, :], axis=0)
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
    last_date = dt.datetime.strptime(find_last_date(), "%Y-%m-%d").date()
    # Assign pool size of the number of processes
    p = multiprocessing.Pool(num_proc)

    # Use functools.partial method to create mapping function
    # for non-simple functions
    # See more here: https://docs.python.org/3/library/functools.html
    mapfunc = partial(process_ticker, datapath=datapath, last_date=last_date, seq_len=seq_len, 
                        target_min=target_min, feats=['Close', 'Volume'], save=True)
    p.map(mapfunc, tickers)

    # Ensure that process has closed and joined before proceeding
    p.close()
    p.join()

    proc_time = (time.time()-t1)
    return proc_time


def combine_seqs(datapath):
    """Combines sequences"""
    # NON SPARK! WORK IN PROGRESS
    files = os.listdir(datapath + 'raw_seq/')
    for i, fn in enumerate(files):
        if i == 0:
            npfiles = np.load(datapath + 'raw_seq/' + fn, allow_pickle=True)
            x_data, y_data = npfiles['x'], npfiles['y']
        else:
            d_add = np.load(datapath + 'raw_seq/' + fn, allow_pickle=True)
            x_add, y_add = d_add['x'], d_add['y']
            x_data = np.concatenate((x_data, x_add), axis=0)
            y_data = np.concatenate((y_data, y_add), axis=0)
    np.savez_compressed(datapath + 'training_data.npz', x_train=x_data, y_train=y_data)
            
if __name__ == '__main__':
    t1 = time.time()
    tickers = read_tickers('all')
    datapath = './'
    if sys.argv[1] != 'all':
        tickers = tickers[:int(sys.argv[1])]
    proctime = process_data_multi(tickers, num_proc=8, datapath='./', seq_len=60, target_min=5, save=True)
    print(f'Processing time: {proctime:0.2f}')
    combine_seqs(datapath)
    print(f'Total time: {time.time()-t1:0.2f}')