
import ccxt
import staticIndicator as SA
import pandas as pd
import datetime



class dataAccess():
    year = 2020
    month = 1
    day = 1
    inquirySize = 240
    simulationDF = pd.DataFrame(columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
    binance = 0
    def __init__(self):
        with open("api.txt") as f:
            lines = f.readlines()
            api_key = lines[0].strip()  # remove special char
            secret = lines[1].strip()

        self.binance = ccxt.binance(config={
            'apiKey': api_key,
            'secret': secret,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future',
                'recvWindow': 60000
            }
        })
  

    # def load_history_data_from_binance():
    def load_history_data_from_binance(self, simDays, unitTime,coinName):
        limitCnt = int(60 * 24 / 60)
        # unitTimeStr = str(unitTime) + 'm'
        unitTimeStr = str(unitTime) + 'h'
        
        
        
        sym = coinName
        

        for day in range(0, simDays):
            dayString = datetime.date(self.year, self.month, self.day) + datetime.timedelta(days=day)
            print(dayString)
            string = str(dayString) + 'T00:00:00Z'
            since = self.binance.parse8601(string)
            coinData = self.binance.fetch_ohlcv(
                symbol=sym,
                since=since,
                # since = None,
                timeframe=unitTimeStr,
                limit=limitCnt)

            one_day_df = pd.DataFrame(coinData, columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
            # if(coinName == self.hedgeCoin):
            #     self.simulationBTDF = pd.concat([self.simulationBTDF, one_day_df], axis=0)
            # else:    
            self.simulationDF = pd.concat([self.simulationDF, one_day_df], axis=0)
        # if(coinName == self.hedgeCoin):
        #     self.simulationBTDF['conv_time'] = pd.to_datetime(self.simulationBTDF['datetime'], unit='ms') + datetime.timedelta(hours=9)
        # else:    
        self.simulationDF = SA.set_rsi_adx_debug(self.simulationDF)
        # self.simulationDF['conv_time'] = pd.to_datetime(self.simulationDF['datetime'], unit='ms') + datetime.timedelta(hours=9)

    

# da = dataAccess()

# simDays = 3
# symbol = 'ETH/USDT'
# unitTime = 1
# da.load_history_data_from_binance(simDays, unitTime ,symbol)

# print(da.simulationDF)


