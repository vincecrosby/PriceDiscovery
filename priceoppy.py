# Predicting Prices
"""
Created on Tue Dec 25 17:41:41 2018

@author: WraithDelta
"""

#import csv
#import datetime as dt 

import pandas as pd
#import pandas_datareader.data as web

import numpy as np

import matplotlib.pyplot as plt
from matplotlib import style
 
from mpl_finance import candlestick_ohlc
import matplotlib.dates as mdates
from matplotlib.dates import date2num

import scipy.fftpack 

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

###########################Grab Stock Lists#####################################
style.use('ggplot')

#stock=input('Input Ticker: ')

############# Load into Pandas DataFrame ######################################
stock_loc = 'stock_dfs/a.csv'
df = pd.read_csv(stock_loc, parse_dates=True, index_col=0)
#print(df.head())

################## Define Moving Averages #####################################
df['5ma'] = df['High'].rolling(window=5, min_periods=5).mean()
df['5ma_std'] = df['Close'].rolling(window=5, min_periods=5).std()

df['10ma'] = df['Close'].rolling(window=10, min_periods=10).mean()
df['10ma_std'] = df['Close'].rolling(window=10, min_periods=10).std()

df['30ma'] = df['Close'].rolling(window=30, min_periods=30).mean()
df['30ma_std'] = df['Close'].rolling(window=30, min_periods=30).std()
df['30ma_var'] = df['Close'].rolling(window=30, min_periods=30).var()

df['200ma'] = df['Close'].rolling(window=200, min_periods=200).mean()
df['200ma_std'] = df['Close'].rolling(window=200, min_periods=200).std()

df['Volume_std'] = df['Volume'].rolling(window=5, min_periods=5).std()






# OHLC Plot handle data 
df_ohlc = df['Close'].resample('1D').ohlc()
df_volume = df['Volume'].resample('1D').sum()

df_ohlc.reset_index(inplace=True)

#Convert datetime object to mdate
df_ohlc['Date'] = df_ohlc['Date'].map(mdates.date2num) 
      
AdjOpen = df['Open']
AdjHigh = df['High']
AdjLow = df['Low']          

#Data handling for candlestick function        
df_ohlc2 = np.array(pd.DataFrame({'0':date2num(df.index.to_pydatetime()),
                                  '1':AdjOpen, '2':AdjHigh, '3':AdjLow,'4':df['Close']}))
#print(df_ohlc2)


################## Exclusion Window ##########################################
#Debt to Equity
#P/E

#Earnings growth
#Letters to Shareholders Flesch Readability Score (<30 Confusing & untrustworthy)

# if high and low is below 10MA
# if the low is < than 2std_dev and low slope > 0
#Trigger buy entry at the close 

#if high is above 10ma
#if the high slope is flat and below zero for at least the previous 3-periods and increases
#trigger buy entry

#Price divergence rate from 10ma (mean reversion)

################## Define Crossover ###########################################
#Define current day condition
daily_close = df['Close']
daily_open = df['Open']
daily_low = df['Low']
daily_high = df['High']

#Define previous day condition
daily_close_prev = df['Close'].shift(1)
daily_open_prev = df['Open'].shift(1)
daily_low_prev = df['Low'].shift(1)
daily_high_prev = df['High'].shift(1)

#Define previous week condition


#Dropping the NaN values
df.dropna(inplace=True) 


#Calculate 2-dayslope high
df['h_slope_chng'] = (daily_high - daily_high_prev)/2
df['hslpma'] = df['h_slope_chng'].rolling(window=7, min_periods=7).mean()

df['hslpstd'] = df['h_slope_chng'].rolling(window=3, min_periods=3).std()

h_up_bound = df['h_slope_chng'].max()
h_low_bound = df['h_slope_chng'].min()
#print(df['hslpma'])

#df['h_wk_slope_chng'] = weekly_high.sub(weekly_high_prev)

#Calculate 2-wk slope
#h_wk_slope_chng = (weekly_high - weekly_high_prev)
#h_wk_slope_chng.dropna(inplace=True)



#print(round(weekly_high_prev,2), round(weekly_high,2), h_wk_slope_chng)

#Calculate 2-dayslope low
df['l_slope_chng'] = (daily_low - daily_low_prev)/2
df['lslpma'] = df['l_slope_chng'].rolling(window=7, min_periods=7).mean()

l_up_bound = df['l_slope_chng'].max()
l_low_bound = df['l_slope_chng'].min()
#print(l_up_bound, l_low_bound)

###### 10/30 Cross ######

df['position'] = df['10ma'] > df['30ma']
df['pre_position'] = df['position'].shift(1)

###### 2/10 Cross #######
df['5maposition'] = df['5ma'] > df['10ma']
df['5mapre_position'] = df['5maposition'].shift(1)

##### 30/200 Cross #######

df['30maposition'] = df['30ma'] > df['200ma']
df['30mapre_position'] = df['30maposition'].shift(1)

#### SLope Cross ####
df['hslp_cross'] = df['h_slope_chng'] > 0
df['prev_hslp_chng'] = df['hslp_cross'].shift(1) 

df.dropna(inplace=True) # dropping the NaN values


df['crossover'] = np.where(df['position'] == df['pre_position'], False, True)
df['5macrossover'] = np.where(df['5maposition'] == df['5mapre_position'], False, True)
df['30macrossover'] = np.where(df['30maposition'] == df['30mapre_position'], False, True)
df['hslpcrossover'] = np.where(df['hslp_cross'] == df['prev_hslp_chng'], False, True)

#Setup plot 
ax1 = plt.subplot2grid((6,1), (1,0), rowspan=4, colspan=1)
ax2 = plt.subplot2grid((6,1), (5,0), rowspan=1, colspan=1, sharex=ax1)
ax0 = plt.subplot2grid((6,1), (0,0), rowspan=1, colspan=4, sharex=ax1)


#############################Long Entry AlgorithmXXXXXXXXXXXXXXXXXXXXXXXXX FIX
#### SLope Cross ####
#print(df['hslpcrossover'])
#print(daily_open)
"""
for i in range(0,len(df['hslpcrossover'])):
    slp_cross = df['hslpcrossover'].values[i] 
    df['buy_sig'] = daily_open.values[i]
    #print(df['buy_sig'])
    
    if slp_cross == True and df['h_slope_chng'].values[i] > 0 :
       # Order(20, 1)
        
        
        ax1.scatter(daily_open.index[i], daily_open.values[i], s = 100, color='yellow', edgecolors='blue', alpha=1.0)
"""

#ax1.xaxis_date()

#PLOT OHLC price candlesticks 
candlestick_ohlc(ax1, df_ohlc2, width=1, colorup='g', colordown='r', alpha=.35)

#PlOT volume data
ax2.fill_between(df_volume.index.map(mdates.date2num), df_volume.values, 0 )
ax2.plot(df.index, df['Volume'], color='black', alpha=0.5)
ax2.plot(df.index, df['Volume']+ df['Volume_std'], color='black', alpha=1)
ax2.bar(df.index, df['Volume'])
   
#PLOT moving averages    

ax1.plot(df.index, df['10ma'], label = '10SMA', color='blue')
ax1.plot(df.index, df['10ma'] + df['10ma_std'], color='blue', linestyle='dashed', alpha=0.35)
ax1.plot(df.index, df['10ma'] - df['10ma_std'], color='blue', linestyle='dashed', alpha=0.35)
ax1.plot(df.index, df['10ma'] + df['10ma_std'] * 2, color='blue', linestyle='dashed', alpha=0.35)
ax1.plot(df.index, df['10ma'] - df['10ma_std'] * 2, color='blue', linestyle='dashed', alpha=0.35)
ax1.plot(df.index, df['10ma'] + df['10ma_std'] * 3, color='blue', linestyle='dashed', alpha=0.35)
ax1.plot(df.index, df['10ma'] - df['10ma_std'] * 3, color='blue', linestyle='dashed', alpha=0.35)
ax1.plot(df.index, df['30ma'], label='30SMA', color='orange')


ax1.plot(df.index, df['200ma'], label= '200SMA', color='purple')

ax1.plot(df.index,df['200ma'] - df['200ma_std'], color='red', alpha=0.25)
ax1.plot(df.index,df['200ma'] + df['200ma_std'], color='red', alpha=0.25)

ax1.plot(df.index,df['200ma'] - (df['200ma_std'] * 2), color='red', alpha=0.25)
ax1.plot(df.index,df['200ma'] + (df['200ma_std'] * 2), color='red', alpha=0.25)

ax1.plot(df.index,df['200ma'] - (df['200ma_std'] * 3), color='red', linestyle='dashed')
ax1.plot(df.index,df['200ma'] + (df['200ma_std'] * 3), color='red', linestyle='dashed')

ax1.legend()
#chartBox1 = ax1.get_position()
#ax1.set_position([chartBox1.x0, chartBox1.y0, chartBox1.width*0.6, chartBox1.height])
#ax1.legend(loc='upper center', bbox_to_anchor=(1.45, 0.8), shadow=True, ncol=1)


"""
for i in range(0,len(df['slope_chng'])):
    diff = df['slope_chng']
   
    if df['slope_chng'].values[i] > 0:
        df['av_pos_diff'] = diff.mean()
        df['std_pos_diff'] = diff.std()
        
    elif df['slope_chng'].values[i] <= 0:
        df['av_neg_diff'] = diff.mean()
        df['std_neg_diff'] = diff.std()
"""        
    
#print(round(df['av_pos_diff'], 2))
#print(round(df['av_neg_diff'], 2))
"""
for i in range(1,len(df['slope_chng'])):
    trend_pos = df.index[i]
  #  print(df['slope_chng'].index[i], round(df['slope_chng'].values[i],2))
    if df['slope_chng'].values[i] > 0.01:
        #df['High'].values[i] +
        trend_mrk =  df['slope_chng'].values[i]
        #print(df['slope_chng'].index[i], round(df['slope_chng'].values[i],2))
        #print(df['slope_chng'].index[i], round(trend_mrk,2))
        
"""
#Calculate  3-period trend change of the highs (- downtrend, + uptrend) 
"""
for i in range(1,len(df['High'])):
    #slp_chng_sum = round(slope_chng.values[i],2).cumsum()
    slp_chng_sum = round(slope_chng.values[i],2)
    print(df.index[i], slp_chng_sum)
slp_pct_chng = slp_chng_sum/df['5ma']*100
"""

#print(round(slope_chng,2))


#Define Price Gap Down characteristics
gap_down_diff = daily_open - daily_close_prev
gap_down_delta = daily_close - daily_low_prev

opp_pct = (gap_down_delta/daily_close_prev)*100

"""
for i in range(1,len(df['Close'])):
    if gap_down_diff.values[i] < 0 and gap_down_delta.values[i] < 0 :
        print(df.index[i],"  ", round(gap_down_diff.values[i], 2), round(gap_down_delta.values[i], 2), round(opp_pct.values[i],1),"%")
       
        ax1.scatter(df.index[i], daily_open.values[i], s = 100, color='yellow', edgecolors='blue', alpha=1.0)

daily_pct_change = daily_close/daily_close.shift(1) - 1
daily_pct_change.fillna(0, inplace=True)
#print(daily_pct_change)
"""
"""
for i in range(1,len(df['crossover'])):
    crosses_10ma = df['crossover'].values[i]
    crosses_5ma = df['5macrossover'].values[i]
    crosses_30ma = df['30macrossover'].values[i]
    
    dpctchange= daily_pct_change.values[i]
#    if daily_close_prev.values[i] > daily_open.values[i] and  
    
    
    
    
    #if crosses_30ma == True and df['30maposition'].values[i] > df['30mapre_position'].values[i] and df['5ma'].values[i] < (df['200ma'].values[i] - (df['200ma'].values[i].std())/4) :
        #ax1.scatter(df['30macrossover'].index[i], df['200ma'].values[i], s = 150, color='green', edgecolors='red')
        #ax1.scatter(df['crossover'].index[i], df['5ma'].values[i], s = 100, color='blue', edgecolors='yellow')
        #pct_diff= (df['5ma'].values[i] - df['10ma'].values[i])/df['5ma'].values[i]
        #print(df['crossover'].index[i], round(pct_diff*100,2),"%", "stdev: ", df['30ma_std'], "var: ", df['30ma_var'] )
      
    if crosses_5ma == True and df['5ma'].values[i] > df['10ma'].values[i] and dpctchange >= 0:
        ax1.scatter(df['crossover'].index[i], df['5ma'].values[i], s = 100, color='yellow', edgecolors='blue', alpha=1.0)
"""   
   # if crosses_10ma == True and df['position'].values[i] > df['pre_position'].values[i]:
   #     ax1.scatter(df['crossover'].index[i], df['10ma'].values[i], s = 100, color='yellow', edgecolors='blue', alpha=1.0)
        
   # if df['5ma'].values[i] < (df['30ma'].values[i] - df['30ma_std'].values[i]) and (df['5maposition'].values[i] > df['5mapre_position'].values[i] ):     
    #   ax1.scatter(df['crossover'].index[i], df['5ma'].values[i], s = 100, color='pink', edgecolors='blue', alpha=1.0)



############################ Plotting #########################################
ax0.plot(df['High'].index, (df['h_slope_chng'].values * 2) **3 , label=' Daily High Slope', color='green', linewidth=0.5, alpha=.8)
ax0.plot(df['Low'].index, (df['l_slope_chng'].values * 2) **3 , label=' Daily Low Slope', color='red', linewidth=0.5, alpha=.8)
ax0.plot(df['hslpma'].index, df['hslpma'].values * 10 , label=' 7-Day MA High Slope', color='blue', linewidth=0.7, alpha=1)
ax0.plot(df['lslpma'].index, df['lslpma'].values * 10 , label=' 7-Day MA Low Slope', color='purple', linewidth=0.7, alpha=1)
ax0.legend()
#ax0.plot(df['h_wk_slope_chng'].index, df['h_wk_slope_chng'].values, color='black', linewidth=0.5, alpha=1)
#ax0.plot(df['Low'].index, df['l_wk_slope_chng'].values, color='green', linewidth=0.5, alpha=1)
#ax0.plot(df['High'].index, df['hslpma'].values, color='purple', linewidth=0.5, alpha=1)



#rsi = rsiFunc(df['Close'])      
#rsiCol = 'b'
posCol = '#386d13'
negCol = '#8f2020'
        
#ax0.plot(df.index, rsi, rsiCol, linewidth=1.5)
ax0.axhline(2, color=negCol, linewidth=0.7)
ax0.axhline(0, color="black", linewidth=0.7, linestyle='dashed')
ax0.axhline(-2, color=posCol, linewidth=0.7)
#ax0.fill_between(df.index, rsi, 90, where=(rsi>=90), facecolor=negCol, edgecolor=negCol, alpha=0.5)
#ax0.fill_between(df.index, rsi, 10, where=(rsi<=10), facecolor=posCol, edgecolor=posCol, alpha=0.5)
ax0.set_yticks([round(h_low_bound,2),round(h_up_bound,2)])
ax0.set_ylim([round(l_low_bound,2),round(l_up_bound,2)])
ax0.yaxis.label.set_color("black")
ax1.yaxis.label.set_color("black")
ax2.yaxis.label.set_color("black")
#ax0.spines['bottom'].set_color("#5998ff")
#ax0.spines['top'].set_color("#5998ff")
#ax0.spines['left'].set_color("#5998ff")
#ax0.spines['right'].set_color("#5998ff")
ax0.tick_params(axis='y', colors='black')
ax0.set_ylabel('Slope Change')
ax1.set_ylabel('Price')
ax2.set_ylabel('Volume')
ax0.tick_params(axis='x', colors='w')
ax1.tick_params(axis='x', colors='w')
ax2.tick_params(axis='x', colors='black', rotation=30)

##############################################################################

plt.title(stock_loc)
plt.show()




#plt.get_backend()



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
"""
account = round(100.00,2)
current_price = round(df['Close'],2)

def Order(cash, shares):
    cash = account - cash
    shares = int(cash/current_price)
    
    order = shares * current_price
    
    if order > account:
        print("Order exceeds account value")
    else:
        order = shares * current_price
        print("Order Executed at: ", current_price,"for ", shares,"shares")    
    
    return Order
 

    
################  Define RSI #################################################
"""
def rsiFunc(prices, n=2):
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
    
