import sys, os, requests, time, math
import datetime as dt
import yfinance as yf
import numpy as np
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
import threading
from functools import partial
import re
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
    dates = np.unique(data_all.index.date)
    for t in tickers:
        for d in dates:
            data_sub = data_all[t].loc[str(d)]
            x = generate_sequences(data_sub, target_min=5, seq_len=60, feats=['Close', 'Volume'])
            y = data_sub['Close'].values[seq_len + target_min:]
            
            # Add to existing sequences
            xs.append(x)
            ys.append(y)
    xs = np.concatenate(xs, axis=0)
    ys = np.concatenate(ys, axis=0)
    return xs, ys

def process_data_seq(tickers, seq_len=60, target_min=5, feats=['Close', 'Volume'], save=True):
    t1 = time.time()
    last_date = dt.datetime.strptime(find_last_date(), "%Y-%m-%d").date()

    # Set variable for today's date
    # Set beginning and end start dates
    start_date = last_date
    end_date = start_date + dt.timedelta(days=7)
    today_dt = dt.datetime.today().date()

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
    np.savez_compressed(datapath + 'training_data.npz', x_train=total_x, y_train=total_y)
    

def process_data_parallel(tickers, n_proc=1, seq_len=60, target_min=5, feats=['Close', 'Volume'], save=True):
    t1 = time.time()
    first_date = dt.datetime.strptime(find_last_date(), "%Y-%m-%d").date()

    # Chunk list of tickers into relatively evenly spaced sizes
    # Note: If there are more processors specified than tickers,
    # make adjustment to require that each processor pull at least
    # two tickers
    if n_proc >= len(tickers):
            size = 2
            n_proc = int(len(tickers)/size)
    else:
        size = math.ceil(len(tickers)/n_proc)
    zs = [tickers[i:i+size] for i in range(0, len(tickers), size)]
    
    # Set variable for today's date
    # Set beginning and end start dates
    start_date = first_date
    end_date = start_date + dt.timedelta(days=7)
    today_dt = dt.datetime.today().date()
    
    total_x, total_y = [], []
    while end_date < today_dt:
        print(start_date, end_date)
        
        # Use functools.partial method to create mapping function
        # for non-simple functions
        # See more here: https://docs.python.org/3/library/functools.html
        mapfunc = partial(process_tickers, start_date=start_date, end_date=end_date, seq_len=seq_len,
                                target_min=target_min, feats=['Close', 'Volume'], save=True)
        with multiprocessing.Pool(n_proc) as p:
            vals = p.map(mapfunc, zs)
            p.close()
            p.join()

        # Concatenate results together into single array
        xs = np.concatenate([v[0] for v in vals], axis=0)
        ys = np.concatenate([v[1] for v in vals], axis=0)

        start_date = end_date + dt.timedelta(days=1)
        end_date += dt.timedelta(days=8)

        total_x.append(xs)
        total_y.append(ys)

    # Convert to single array and save out
    total_x = np.concatenate(total_x, axis=0)
    total_y = np.concatenate(total_y, axis=0)
    np.savez_compressed(datapath + 'training_data.npz', x_train=total_x, y_train=total_y)    
        


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
        x = generate_sequences(data, target_min=target_min, seq_len=seq_len, feats=feats)
        y = data['Close'].values[seq_len + target_min:]

        xs.append(x)
        ys.append(y)

    xs = np.concatenate(xs, axis=0)
    ys = np.concatenate(ys, axis=0)
    try:
        os.mkdir( datapath + 'raw_seq')
    except:
        pass

    np.savez_compressed(datapath + 'raw_seq/' + t + '.npz', x=xs, y=ys)

def process_data_mp(tickers, num_proc=None, datapath='./', seq_len=60, target_min=5, save=True):
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

            
if __name__ == '__main__':
    t1 = time.time()
    tickers = read_tickers('all')
    datapath = './'
    if sys.argv[1] != 'all':
        tickers = tickers[:int(sys.argv[1])]

    
    print(f'Total time: {time.time()-t1:0.2f}')