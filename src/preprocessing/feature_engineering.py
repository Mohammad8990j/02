import talib as ta
import pandas as pd

def calculate_indicators(df):
    # محاسبه اندیکاتورها
    df['RSI'] = ta.RSI(df['close'], timeperiod=14)
    df['MA'] = ta.SMA(df['close'], timeperiod=50)
    df['MACD'], df['MACD_signal'], df['MACD_hist'] = ta.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
    df['MFI'] = ta.MFI(df['high'], df['low'], df['close'], df['volume'], timeperiod=14)

    # بازگشت دیتافریم با اندیکاتورها
    return df
