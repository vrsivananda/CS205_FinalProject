# Author: Kevin Hare
# Last Updated: 5/1/2021
# Purpose: Test performance for weak scaling and strong scaling
import numpy as np
from process_prices import find_last_date, read_tickers, process_data_parallel, process_data_seq, generate_sequences
import sys, os, re, time
import datetime as dt
import yfinance as yf
import multiprocessing
from functools import partial
sys.path.append('..')


def full_seq_proc(tickers, datapath='./', seq_len=60, target_min=5, save=True):
    """FULLY SEQUENTIAL VERSION!
    Inputs
    ------
    tickers: iterable, ticker symbols to iterate over, required
    datapath: path to data storage files
    seq_len: (optional) length of sequence in minutes
    target_min: (optional) target minutes ahead (default set at 5 min)"""
    t1 = time.time()
    today = dt.date.today()
    last_date = dt.datetime.strptime(find_last_date(), "%Y-%m-%d").date()
    xt, yt = [], []
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

        #np.savez(datapath + 'raw_seq/' + t + '.npz', x=xs, y=ys)
        xt.append(xs)
        yt.append(ys)
    # Return processing time
    return time.time()-t1


def proc_single_ticker(t, datapath, last_date, seq_len=60, target_min=5, feats=['Close', 'Volume'], save=True):
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

    #np.savez_compressed(datapath + 'raw_seq/' + t + '.npz', x=xs, y=ys)
    return xs, ys

def process_data_mp(tickers, num_proc=None, datapath='./', seq_len=60, target_min=5, save=True):
    """Applies multiprocessing to tickers fed into a ticker list, wraps the process ticker, 
    uses functools.partial() to apply closure pattern to single ticker processing

    Note: Used functools.partial() rather than self-closure for optimization purposes
    
    Returns
    -------
    proc_time: time in seconds to perform processing"""
    t1 = time.time()
    last_date = dt.datetime.strptime(find_last_date(), "%Y-%m-%d").date()

    # Use functools.partial method to create mapping function
    # for non-simple functions
    # See more here: https://docs.python.org/3/library/functools.html
    mapfunc = partial(proc_single_ticker, datapath=datapath, last_date=last_date, seq_len=seq_len, 
                        target_min=target_min, feats=['Close', 'Volume'], save=True)
    with multiprocessing.Pool(num_proc) as p:
        vals = p.map(mapfunc, tickers)
        # Ensure that process has closed and joined before proceeding
        p.close()
        p.join()

    xs = np.concatenate([v[0] for v in vals], axis=0)
    ys = np.concatenate([v[1] for v in vals], axis=0)

    proc_time = (time.time()-t1)
    return proc_time

def initial_perf_scaling_test(x):
    """Compares fully sequential version of code to 
    first level of parallelism"""
    tickers = read_tickers('all')[:x]
    datapath = './'
    seq_time = full_seq_proc(tickers, datapath='./', seq_len=60, target_min=5, save=True)
    t1 = time.time()
    process_data_seq(tickers, save=False)
    new_seq_time = time.time()-t1
    init_par_times = []
    new_par_times = []
    procs = [1, 2, 4, 8]
    for p in procs:
        ptime = process_data_mp(tickers, num_proc=p, save=False)
        init_par_times.append(ptime)
        t1 = time.time()
        process_data_parallel(tickers, n_proc=p, save=False)
        nt = time.time()-t1
        new_par_times.append(nt)
    
    with open('./initial_performance_testing.txt', 'w') as f:
        f.write('Sequential Time: \n')
        f.write(str(seq_time))

        init_par_times = [str(x) + '\t' for x in init_par_times]
        f.write('\nInitial Parallel Times: \n')
        f.writelines(init_par_times)

        f.write('\nNew Sequential Time:\n')
        f.write(str(new_seq_time))

        new_par_times = [str(x) + '\t' for x in new_par_times]
        f.write('\nNew Parallel Times: \n')
        f.writelines(new_par_times)

        f.close()
    
def strong_scaling(n=100):
    """Demonstrates strong scaling performance enhancements
    from sequential time and multiprocessing
    """
    # Load in consistent set of tickers
    tickers = read_tickers('all')[:n]
    
    # timing function
    def timing_fn(fn, tickers, **kwargs):
        t1 = time.time()
        fn(tickers, **kwargs)
        return time.time()-t1

    seq_time = timing_fn(process_data_seq, tickers, save=True)
    par_times = []
    procs = [1, 2, 4, 8]
    for p in procs:
        p_time = timing_fn(process_data_parallel, tickers, n_proc=p, save=True)
        par_times.append(p_time)
    
    with open('./strong_scaling.txt', 'w') as f:
        f.write('Sequential Time: \n')
        f.write(str(seq_time))

        par_times = [str(x) + '\t' for x in par_times]
        f.write('\Parallel Time: \n')
        f.writelines(par_times)
        f.close()


if __name__ == '__main__':
    #initial_perf_scaling_test(int(sys.argv[1]))
    #strong_scaling(n=1)
    print('hello')
