import talib
import numpy as np
import pandas as pd

# This function is to set statical data for release version.



def set_rsi_adx(df):
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values

    df['RSI'] = talib.RSI(close, timeperiod=6) # / tb.RSI(c, timeperiod=6).mean()
    df['ADX'] = talib.ADX(high, low, close, timeperiod=6) # / tb.ADX(high, low, close, timeperiod=6).mean()
    df['5MA'] = talib.MA(close, timeperiod=5, matype=0) # / tb.ADX(high, low, close, timeperiod=6).mean()
    df['10MA'] = talib.MA(close, timeperiod=10, matype=0) # / tb.ADX(high, low, close, timeperiod=6).mean()
    df['20MA'] = talib.MA(close, timeperiod=20, matype=0) # / tb.ADX(high, low, close, timeperiod=6).mean()
    df['60MA'] = talib.MA(close, timeperiod=60, matype=0) # / tb.ADX(high, low, close, timeperiod=6).mean()
    # df['ADX'] = talib.ADX(high, low, close, timeperiod=6) # / tb.ADX(high, low, close, timeperiod=6).mean()
    upperband, middleband, lowerband = talib.BBANDS(close, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)
    df['BBUP'] = upperband #가격의 표준편차로 주로 상방추세 볼린저밴드상방에 있는경우 중간으로 내려오려는 성격이 있음
    df['BBMID'] = middleband
    df['BBLOW'] = lowerband
    df['CMO'] = talib.CMO(close, timeperiod=9) # CMO가 0기준으로 -50에 가까울수록 하단추세 50에 가까울수록 상단추세
    df['SAR'] = talib.SAR(high, low, acceleration=0.02, maximum=0.2) # 파라볼릭 캔들 아래 위치한경우 하락 위에 위치한 경우 상방
    macd, macdsignal, macdhist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
    df['macd'] = macd
    df['macdhist'] = macdhist
    df['ADX14'] = talib.ADX(high, low, close, timeperiod=14) # / tb.ADX(high, low, close, timeperiod=6).mean()
    df['PDI'] = talib.PLUS_DI(high, low,close, timeperiod=14)
    df['MDI'] = talib.MINUS_DI(high, low,close, timeperiod=14)
    return df

# This function is to set statical data for debug.
def set_rsi_adx_debug(df):
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    df['RSI'] = talib.RSI(close, timeperiod=6) # / tb.RSI(c, timeperiod=6).mean()
    df['RSI14'] = talib.RSI(close, timeperiod=14)  # / tb.RSI(c, timeperiod=6).mean()
    df['CMO'] = talib.CMO(close, timeperiod=9) # CMO가 0기준으로 -50에 가까울수록 하단추세 50에 가까울수록 상단추세
    df['5MA'] = talib.MA(close, timeperiod=5, matype=0) # / tb.ADX(high, low, close, timeperiod=6).mean()
    df['10MA'] = talib.MA(close, timeperiod=10, matype=0) # / tb.ADX(high, low, close, timeperiod=6).mean()
    df['ADX'] = talib.ADX(high, low, close, timeperiod=6) # / tb.ADX(high, low, close, timeperiod=6).mean()
    df['ADX14'] = talib.ADX(high, low, close, timeperiod=14) # / tb.ADX(high, low, close, timeperiod=6).mean()
    df['PDI'] = talib.PLUS_DI(high, low,close, timeperiod=14)
    df['MDI'] = talib.MINUS_DI(high, low,close, timeperiod=14)
    macd, macdsignal, macdhist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
    df['macd'] = macd
    df['macdhist'] = macdhist
    upperband, middleband, lowerband = talib.BBANDS(close, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)
    df['BBUP'] = upperband #가격의 표준편차로 주로 상방추세 볼린저밴드상방에 있는경우 중간으로 내려오려는 성격이 있음
    df['BBMID'] = middleband
    df['BBLOW'] = lowerband
    
    return df


