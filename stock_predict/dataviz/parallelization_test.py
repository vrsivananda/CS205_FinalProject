import sys, os, requests, time
import datetime as dt
import yfinance as yf
import numpy as np
import multiprocessing
from functools import partial
from process_prices import find_last_date, read_tickers, process_tickers, process_ticker, process_data_multi
import matplotlib.pyplot as plt


def gen_times(tickers, proc_num_list):
    times = []
    for i, proc in enumerate(proc_num_list):
        t = process_data_multi(tickers, num_proc=proc, datapath='./', seq_len=60, target_min=5, save=True)
        times.append(t)
    return times

def plot_times(serial_time, parallel_times):
    fig = plt.figure()
    t_comb = [serial_time] + parallel_times
    l = len(t_comb)
    plt.plot(np.arange(l), t_comb)
    plt.savefig('parallel_times.png')
    
def plot_speedup(serial_time, parallel_times, proc_list):
    """Plots speedup for parallel times based on number of available processes (cores)"""
    fig = plt.figure()
    speedup = [serial_time/p for p in parallel_times]
    plt.plot(proc_list, speedup, label='Actual speedup')
    plt.plot(proc_list, proc_list, label='Theoretical speedup', ls='--', c='k')
    plt.title('Benchmark performance speedup for data download')
    plt.xlabel('Number of processes')
    plt.ylabel('Speedup')
    plt.legend()
    plt.savefig('data_proc_speedup.png')

    

if __name__ == '__main__':
    tickers = read_tickers('all')
    datapath = './'
    procs = [1, 2, 4, 8]
    st = process_tickers(tickers[:5], datapath)
    pt = gen_times(tickers[:5], procs)
    plot_speedup(st, pt, procs)

