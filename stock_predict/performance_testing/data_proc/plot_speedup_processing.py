import numpy as np
import matplotlib.pyplot as plt
import os, sys, re

def get_times(filename):
    sequential_times = []
    parallel_times = []
    with open('./' + str(filename), 'r') as f:
        for l in f:
            if re.match('[0-9]', l):
                d = l.strip().split('\t')
                if len(d) == 1:
                    sequential_times.append(d[0])
                else:
                    parallel_times.append(d)
    
    return sequential_times, parallel_times


def plot_speedup_singlethread():
    """Plots speedup for a single thread, 
    using multiple processors. Highlights theoretical performance"""
    # Read in .txt using helper method outlined above
    sequential_times, parallel_times = get_times('initial_performance_testing.txt')
    fig, axs = plt.subplots(1, 1, figsize=(6,4))
    speedup_init = [float(sequential_times[0])/float(p) for p in parallel_times[0]]
    speedup_par_thread = [float(sequential_times[0])/float(sequential_times[1]) for p in parallel_times[0]]
    speedup_par_all = [float(sequential_times[0])/float(p) for p in parallel_times[1]]
    speedup_par_threadbase = [float(sequential_times[1])/float(p) for p in parallel_times[1]]

    # Note: Just plot multicore vs multithread + multicore
    cores = [1,2,4,8]
    axs.plot(cores, speedup_init, label=f'Single threading')
    axs.set_xlabel('Number of processes')
    axs.set_ylabel('Speedup')
    axs.legend()
    plt.suptitle(f'Speedup for multiple processes, single thread per process (50 tickers)')
    plt.legend()
    
    plt.savefig('./speedup_multithreading.png')
    plt.show()


def plot_relative_multithread_gain():
    """Plots the relative gains of using multithreaded approach, holding constant
    the number of processors.

    Shows the relative efficiency of multithreading and efficiency"""
    sequential_times, parallel_times = get_times('initial_performance_testing.txt')
    fig, axs = plt.subplots(1, 1, figsize=(6,4))
    speedup_init = [float(sequential_times[0])/float(p) for p in parallel_times[0]]
    speedup_par_thread = [float(sequential_times[0])/float(sequential_times[1]) for p in parallel_times[0]]
    speedup_par_all = [float(sequential_times[0])/float(p) for p in parallel_times[1]]
    speedup_par_threadbase = [float(sequential_times[1])/float(p) for p in parallel_times[1]]   

    # Calculate efficiency
    cores = [1,2,4,8]
    rel_speedup = [float(x)/float(y) for x, y in zip(parallel_times[0], parallel_times[1])]
    plt.plot(cores, rel_speedup, label=f'Speedup')
    plt.xlabel('# of Processes')
    plt.ylabel('Relative Speedup')
    plt.title('Relative speedup of multithreaded approach, by number of processes (50 tickers)')
    plt.ylim(0, 8)

    plt.show()

def plot_speedup_all():
    """Plots speedup; assumes sequential updated version
    will be same across cores"""
    sequential_times, parallel_times = get_times('initial_performance_testing.txt')
    fig, axs = plt.subplots(1, 2, figsize=(10,6))
    speedup_init = [float(sequential_times[0])/float(p) for p in parallel_times[0]]
    speedup_par_thread = [float(sequential_times[0])/float(sequential_times[1]) for p in parallel_times[0]]
    speedup_par_all = [float(sequential_times[0])/float(p) for p in parallel_times[1]]
    speedup_par_threadbase = [float(sequential_times[1])/float(p) for p in parallel_times[1]]

    cores = [1,2,4,8]
    axs = axs.flatten()
    axs[0].plot(cores, speedup_init, label=f'Single threading, multi core')
    axs[0].plot(cores, speedup_par_thread, label=f'Speedup with multithreading, single core')
    axs[0].plot(cores, speedup_par_all, label=f'Speedup with multithreading, multi core')
    axs[0].set_xlabel('Processors')
    axs[0].set_ylabel('Speedup')
    axs[0].legend()

    axs[1].plot(cores, speedup_par_threadbase, label=f'Speedup over multithread, single core')
    axs[1].set_xlabel('Processors')
    axs[1].set_ylabel('Speedup')
    axs[1].set_title('Comparison using with multithreading baseline')
    plt.suptitle(f'Plot using 50 tickers')
    plt.legend()
    
    plt.savefig('./speedup_processing_all.png')
    plt.show()

def plot_speedup_multithread():
    """Only plots multithreading speedup"""
    sequential_times, parallel_times = get_times('strong_scaling.txt')
    fig, axs = plt.subplots(1, 1, figsize=(6,4))
    sequential_times = re.findall('[0-9\.]*', sequential_times[0])[0]
    speedups = [float(sequential_times)/float(p) for p in parallel_times[0]]
    cores = [1,2,4,8]
    plt.plot(cores, speedups, label=f'Actual Speedup')
    plt.plot(cores, cores, ls='--', c='k', label=f'Predicted Speedup')
    plt.xlabel('Processors')
    plt.ylabel('Speedup')
    plt.title('Speedup with multiple processes (500 tickers)')
    plt.legend()
    plt.savefig('./full_multithread_perf.png')
    plt.show()


def plot_speedup_total():
    """Plots the total speedup vs. a naive process"""
    sequential_times, parallel_times = get_times('initial_performance_testing.txt')
    fig, axs = plt.subplots(1, 1, figsize=(6,4))
    speedup_par_all = [float(sequential_times[0])/float(p) for p in parallel_times[1]]

    cores = [1,2,4,8]
    axs.plot(cores, speedup_par_all, label=f'Total Speedup')
    axs.set_ylabel('Speedup')
    axs.set_xlabel('Processors')
    axs.set_title('Speedup over naive baseline')

    plt.legend()
    plt.savefig('./speedup_all.png')
    plt.show() 


if __name__ == '__main__':
    plot_speedup_singlethread()
    plot_speedup_all()
    plot_speedup_multithread()
    plot_relative_multithread_gain()
    plot_speedup_total()
    




