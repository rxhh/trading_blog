import sys
import datetime as dt
import io
import os
import pandas as pd
import requests
import zipfile

def download_klines_on_date(symbol, d):
    filestr = f"{symbol}_UMCBL_1min_{d.strftime('%Y%m%d')}"
    url = f"https://img.bitgetimg.com/online/kline/{symbol}/{filestr}.zip"
    r = requests.get(url)
    try:
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extract(f"{filestr}.xlsx", f"../temp")
        df = pd.read_excel(f"../temp/{filestr}.xlsx")
        df = df.set_index(pd.to_datetime(df['timestamp'], unit='s', utc=True).rename('dt'))
        df.columns = df.columns.str.lower()
        df = df.rename(columns={'basevolume':'volume'})
        df.to_parquet(f"../data/candles/{symbol}/{d.strftime('%Y%m%d')}.pq")
        return True
    except:
        return False

def download_klines_in_date_range(symbol, start, end):
    for d in pd.date_range(start, end).date:
        result = download_klines_on_date(symbol, d)
        if result:
            print(f"Successfully downloaded klines for {d}")
        else:
            print(f"Error on {d}")
    return

def load_klines_in_date_range(symbol, start, end):
    ddfs = []

    for d in pd.date_range(start, end).date:
        try:
            df = pd.read_parquet(f"../data/candles/{symbol}/{d.strftime('%Y%m%d')}.pq")
            ddfs.append(df)
        except:
            print(f"Error on {d}")
    return pd.concat(ddfs).sort_index()