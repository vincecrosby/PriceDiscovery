# Predicting Prices
"""
Created on Tue Dec 25 17:41:41 2018

@author: WraithDelta
"""
#import csv
#import numpy as np

import datetime as dt 

import pandas as pd
import pandas_datareader.data as web

import numpy as np
 


import matplotlib.pyplot as plt
from matplotlib import style
 
from mpl_finance import candlestick_ohlc
import matplotlib.dates as mdates

#from sklearn.svm import SVR

###########################Grab Stock Lists#####################################


###############################################################################
#pd.options.display.max_rows = 10

style.use('ggplot')

#start = dt.datetime(2000, 1, 1)
#end = dt.datetime(2016, 12, 31)

#df = web.DataReader('TSLA', 'yahoo', start, end)
#print(df.head(8))
#df.to_csv('tsla.csv')

############# Load into Pandas DataFrame ######################################

df = pd.read_csv('tsla.csv', parse_dates=True, index_col=0)

#print(df.head())


################## Define Moving Averages #####################################

df['10ma'] = df['Adj Close'].rolling(window=10, min_periods=10).mean()
df['30ma'] = df['Adj Close'].rolling(window=30, min_periods=30).mean()
df['200ma'] = df['Adj Close'].rolling(window=200, min_periods=200).mean()

df_ohlc = df['Adj Close'].resample('5D').ohlc()
df_volume = df['Volume'].resample('5D').sum()

df_ohlc.reset_index(inplace=True)
df_ohlc['Date'] = df_ohlc['Date'].map(mdates.date2num) #convert datetime object to mdate

#print(df_ohlc.head())

################## Define Crossover ###########################################

df['position'] = df['10ma'] > df['30ma']
df['pre_position'] = df['position'].shift(1)
df.dropna(inplace=True) # dropping the NaN values
df['crossover'] = np.where(df['position'] == df['pre_position'], False, True)

#print (df['crossover'].index[0])




############################ Plotting #########################################


ax1 = plt.subplot2grid((6,1), (0,0), rowspan=5, colspan=1)
ax2 = plt.subplot2grid((6,1), (5,0), rowspan=1, colspan=1, sharex=ax1)
ax1.xaxis_date()

candlestick_ohlc(ax1, df_ohlc.values, width=2, colorup='g')
ax2.fill_between(df_volume.index.map(mdates.date2num), df_volume.values, 0 )

#ax1.plot(df.index, df['Adj Close'])
ax1.plot(df.index, df['10ma'], color='cyan')
ax1.plot(df.index, df['30ma'], color='orange')
ax1.plot(df.index, df['200ma'], color='purple')

########################## PLot where positive crossover occurs ########################
for i in range(1,len(df['crossover'])):
    crosses = df['crossover'].values[i]
    if crosses == True and df['position'].values[i] > df['pre_position'].values[i]:
        cross_values = df['10ma'].values[i]
       # print (df['crossover'].index[i])
       # print(cross_values)
        ax1.scatter(df['crossover'].index[i], df['10ma'].values[i], s = 100, color='blue', edgecolors='red')


ax2.bar(df.index, df['Volume'])

plt.title('TSLA')
plt.xlabel('Date')
plt.xticks(rotation=90)
plt.ylabel('Price')
#plt.legend()
plt.show()


#df.plot('Close')
#plt.show

#plt.get_backend()

date_data = []
open_data = []
high_data = []
low_data = []
close_data = []
adjclose_data = []
volume_data = []

"""
################Gets data from csv file and parses it##########################
def get_data(filename):
    with open(filename, 'r') as csvfile:
        csvFileReader = csv.reader(csvfile)
        next(csvFileReader)
        
        for row in csvFileReader:
            
            date_data.append(str(row[0].split(' ')[0]))
            print(date_data)
            open_data.append(float(row[1]))
            high_data.append(float(row[2]))
            low_data.append(float(row[3]))
            close_data.append(float(row[4]))
            adjclose_data.append(float(row[5]))
            volume_data.append(int(row[6]))
            
      #  print (dates, openn, high, low, close, adjclose)       
    return
###############################################################################
   

    
def rsiFunc(prices, n=14):
    deltas = np.diff(prices)
    seed = deltas[:n+1]
    up = seed[seed>0].sum()/n
    down = -seed[seed<0].sum()/n
    rs = up/down
    rsi = np.zeros_like(prices)
    rsi[:n] = 100. - 100./(1.+rs)
    
    for i in range(n, len(prices)):
        delta = deltas[i-1]     
    
        if delta > 0:
           upval = delta
           downval = 0.
        else:
           upval = 0.
           downval = -delta
        up = (up*(n-1) + upval) /n
        down = (down*(n-1)+downval)/n
        rs = up/down
        rsi[i] = 100. - 100./(1.+rs)
    return rsi   
""" 
           

###############################################################################

#def predict_prices(dates, high, x):
#    dates = np.reshape(dates,(len(dates), 1))
    
    
    
#    svr_lin = SVR(kernel= 'linear', C=1e3)
#    svr_poly = SVR(kernel= 'poly', C=1e3, degree = 2)
#    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
    
#    svr_rbf.fit(dates, high) # fitting the data points in the models
#    svr_lin.fit(dates, high)
#    svr_poly.fit(dates, high)
    
#    plt.scatter(dates, high, color='black', label='Data')
#    plt.plot(dates, high, color='pink', label='Data')

#    plt.plot(svr_rbf.predict(dates), color='red', label='RBF model')
#    plt.plot(svr_lin.predict(dates), color='green', label='Linear model')
#    plt.plot(svr_poly.predict(dates), color='blue', label='Polynomial model')
   
#    plt.xlabel('Date')
#    plt.ylabel('Price')
#    plt.title('Support Vector Regression')
    
#    plt.legend()
#    plt.show()
    
#    return svr_rbf.predict(x)[0], svr_lin.predict(x)[0], svr_poly.predict(x)[0]
###############################################################################
    
#get_data('AAPL.csv')
#rsi = rsiFunc(close_data) 


#predicted_price = predict_prices(dates, high_data, [[29]])
#print (predicted_price)
#plt.plot(date_data, rsi, color='blue', label='RSI')
#plt.scatter(date_data, high_data, color='black', label='highData')
#plt.plot(date_data, high_data, color='pink', label='highData')
#plt.xlabel('Date')
#plt.ylabel('Price')
#plt.title('AAPL Highs')

#plt.grid()    
#plt.legend()
#plt.show()
    