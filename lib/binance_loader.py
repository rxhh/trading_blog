import os
import ccxt
import datetime as dt
import numpy as np
import pandas as pd

def download_ohlc_binance(symbol, start):
    binance = ccxt.binance({ 'options': {'defaultType': 'future' }})
    start_dt = dt.datetime.combine(start, dt.time(0))
    path_to = f"../data/candles/{symbol}.pq"
    if not os.path.isfile(path_to):
        all_klines = []
        params = {
            'interval': '1m',
            'limit': 1000,
            'symbol': symbol,
        }
        klines = pd.DataFrame(binance.fapiPublicGetKlines(params))
        all_klines.append(klines)
        t0 = pd.to_datetime(int(klines[0].iloc[0]), unit='ms')
        while t0 > start_dt:
            params['endTime'] = klines[0].iloc[0]
            klines = pd.DataFrame(binance.fapiPublicGetKlines(params))
            all_klines.append(klines)
            _t0 = pd.to_datetime(int(klines[0].iloc[0]), unit='ms')
            if t0 == _t0: break
            else: t0 = _t0
        df = parse_binance_ohlc(pd.concat(all_klines)).iloc[:-1]
        df.to_parquet(path_to)
    else:
        df = pd.read_parquet(path_to)
        
        # Forward fill to present    
        last_dt = df.index[-1]
        all_klines = []
        params = {
            'interval': '1m',
            'limit': 1000,
            'symbol': symbol,
        }
        klines = pd.DataFrame(binance.fapiPublicGetKlines(params))
        all_klines.append(klines)
        t0 = pd.to_datetime(int(klines[0].iloc[0]), unit='ms')
        while t0 > last_dt:
            params['endTime'] = klines[0].iloc[0]
            klines = pd.DataFrame(binance.fapiPublicGetKlines(params))
            all_klines.append(klines)
            t0 = pd.to_datetime(int(klines[0].iloc[0]), unit='ms')
        df = pd.concat([parse_binance_ohlc(pd.concat(all_klines)).iloc[:-1], df]).drop_duplicates().sort_index()    
        
        # Backfill
        if start_dt < df.index[0]:
            t0 = df.index[0]
            ts0 = df['timestamp'].iloc[0]
            all_klines = []
            while t0 > start_dt:
                params = {
                    'interval': '1m',
                    'limit': 1000,
                    'symbol': symbol,
                    'endTime': ts0
                }
                klines = pd.DataFrame(binance.fapiPublicGetKlines(params))
                all_klines.append(klines)
                _t0 = pd.to_datetime(int(klines[0].iloc[0]), unit='ms')
                ts0 = klines[0].iloc[0]
                if t0 == _t0: break
                else: t0 = _t0
            df = pd.concat([parse_binance_ohlc(pd.concat(all_klines)), df]).drop_duplicates().sort_index()
            df.to_parquet(path_to)
    return

def parse_binance_ohlc(df):
    df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quoteVolume', 'nTrades', 'upVolume', 'upQuoteVolume', 'ignore']
    df = df.drop(columns='ignore')
    df = pd.concat([df[['timestamp', 'close_time']].astype(np.int64), df[['open', 'high', 'low', 'close', 'volume', 'quoteVolume', 'nTrades', 'upVolume', 'upQuoteVolume']].astype(float)], axis=1)
    df = df.drop_duplicates()
    df = df.set_index(pd.to_datetime(df['timestamp'].astype(np.int64), unit='ms')).sort_index()
    df.index.name = 'dt'
    return df

def load_ohlc_in_date_range(symbol, start, end):
    path_to = f"../data/candles/{symbol}.pq"
    if not os.path.isfile(path_to):
        print(f"{symbol} not found")
    else:
        return pd.read_parquet(path_to).loc[start:end]