# Predicting Prices
"""
Created on Tue Dec 25 17:41:41 2018

@author: WraithDelta
"""
import csv
import numpy as np

import datetime as dt 

import pandas as pd
import pandas_datareader.data as web
 
from sklearn.svm import SVR

import matplotlib.pyplot as plt
from matplotlib import style 

style.use('ggplot')

start = dt.datetime(2000, 1, 1)
end = dt.datetime(2016, 12, 31)

df = web.DataReader('TSLA', 'yahoo', start, end)
#print(df.head(8))
df.to_csv('tsla.csv')
df = pd.read_csv('tsla.csv', parse_dates=True, index_col=0)

df.plot('High')
#plt.show


plt.get_backend()

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
"""    
    
    
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
    