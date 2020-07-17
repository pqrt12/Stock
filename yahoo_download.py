#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import os
from datetime import datetime
#from bs4 import BeautifulSoup
#from IPython.display import display_html

from config import final_download_dir, DATETIME_FORMAT

# =================================================================
# Yahoo historical data download specific
# this thing works !!!
def get_yahoo_timestamp(dt):
    t = datetime.timestamp(dt) + 61200.0
    return int(t)

def get_yahoo_url(ticker, start_dt, stop_dt):
    # constant
    web_url_prefix = 'https://query1.finance.yahoo.com/v7/finance/download/'
    web_url_suffix = '&interval=1d&events=history'
    # template
    start = get_yahoo_timestamp(start_dt)
    stop = get_yahoo_timestamp(stop_dt)
    template = ticker.upper() + '?period1=' + str(start) + '&period2=' + str(stop)

    # web_url_prefix:  'https://finance.yahoo.com/quote/'
    # template:        'VIG/history?period1=1436140800&period2=1593993600'
    # web_url_suffix:  '&interval=1d&filter=history&frequency=1d'
    return web_url_prefix + template + web_url_suffix

def get_yahoo_hist_df(ticker,
                start_str='1990-01-01',
                stop_str=datetime.today().strftime(DATETIME_FORMAT)):
    # format to yahoo finance url.
    web_url = get_yahoo_url(ticker,
                    str2datetime(start_str),
                    str2datetime(stop_str))

    return pd.read_csv(web_url)

def dnld_yahoo_hist_data(ticker,
                start_str='1990-01-01',
                stop_str=datetime.today().strftime(DATETIME_FORMAT)):
    # visit the webpage
    df = get_yahoo_hist_df(ticker, start_str, stop_str)

    # move the file to the final location (replace works in case exist already)
    dnld_filename = ticker.upper() + '.csv'
    final_file = os.path.join(final_download_dir, dnld_filename)
    df.to_csv(final_file, index=False)

    # for visual check only, may comment out.
    display(df.head(3))

# example:
# earlier start-time is ok.
# dnld_yahoo_hist_data('vgt', '1980-01-01', '2020-07-05')
# input in string
def str2datetime(date_str):
    return datetime.strptime(date_str, DATETIME_FORMAT)

def import_yahoo_df(ticker):
    return get_yahoo_hist_df(ticker)
