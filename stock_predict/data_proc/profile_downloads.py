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
    


if __name__ == '__main__':
    tickers = read_tickers('all')
    datapath = './'
    #process_tickers(tickers[:5], datapath)
    pull_data_together(tickers[:5])

    