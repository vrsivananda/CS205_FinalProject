import numpy as np
import yfinance as yf
from process_prices import  process_tickers, read_tickers, find_last_date
import datetime as dt
import sys
sys.path.append('../')


def pull_data_together(tickers):
    today = dt.date.today()
    last_date = dt.datetime.strptime(find_last_date(), "%Y-%m-%d").date()
    day = last_date
    while day < today:
        print(day)
        end_day = day + dt.timedelta(days=1)
        x = yf.download(tickers, period='1d', interval='1m', start=str(day), end=str(end_day), progress=False)
        day += dt.timedelta(days=1)

def pull_data_together_seq(tickers):
    today = dt.date.today()
    last_date = dt.datetime.strptime(find_last_date(), "%Y-%m-%d").date()
    day = last_date
    for t in tickers:
        while day < today:
            print(day)
            end_day = day + dt.timedelta(days=1)
            x = yf.download(t, period='1d', interval='1m', start=str(day), end=str(end_day), progress=False)
            day += dt.timedelta(days=1)
        day = last_date
    


if __name__ == '__main__':
    tickers = read_tickers('all')
    datapath = './'
    if sys.argv[2] == 'parallel':
        pull_data_together(tickers[:int(sys.argv[1])])
    else:
        print(tickers[:int(sys.argv[1])])
        pull_data_together_seq(tickers[:int(sys.argv[1])])

    