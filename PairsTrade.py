from pathlib import Path
import glob
import pandas as pd
import os
import glob
import pandas as pd
import seaborn as sb
import numpy as np
from numpy import cov
import datetime
import matplotlib.pyplot as plt
path = r'/Users/danielchen/Desktop/Historical_data/stockpicks/*.csv'
files = glob.glob(path)
dfs = [pd.read_csv(f, index_col = None, header = 0, sep=',' ,usecols = ['close']) for f in files]

stock_1 = []
stock_2 = []
cor_list = []
for i in range (0,len(dfs)):
    df1 = dfs[i]
    for j in range (1,len(dfs)):
        df2 = dfs[j]
        correlation = float((df1.corrwith(df2, axis=0)))
        if i == j:
            continue
        if correlation > 0.8:
            stock_1.append(i)
            stock_2.append(j)
            cor_list.append(correlation)
highest_cor = (cor_list.index(max(cor_list)))
i_index = stock_1[highest_cor]
j_index = stock_2[highest_cor]
stock1 = str(files[stock_1[highest_cor]])
stock2 = str(files[stock_2[highest_cor]])
stock1_csv = str(stock1[59:])
stock2_csv = str(stock2[59:])
print("Highest Correlation Stocks are " + stock1_csv + " and " + stock2_csv + " with a correlation of " + str(max(cor_list)))
fulldata = pd.concat([dfs[i_index], dfs[j_index]])
#print(fulldata)

S1 = dfs[i_index]
Stock_1 = S1.iloc[::-1]
S2 = dfs[j_index]
Stock_2 = S2.iloc[::-1]

ratios = Stock_1 / Stock_2
#print(ratios)
def zscore(fulldata):
    return (fulldata - fulldata.mean()) / np.std(fulldata)
zscore_df = (fulldata - fulldata.mean()) / np.std(fulldata)
#zscore plot
zscore(ratios).plot(figsize=(14,7))
plt.axhline(zscore(ratios["close"]).mean())
plt.axhline(1.0, color='red')
plt.axhline(-1.0, color='green')
plt.legend(['Z-Score'])
plt.show()

train = ratios[:7908]
test = ratios[7908:]

#Various Time Ratios Plot
ratios_mavghourly = train.rolling(window=60, center=False).mean()
ratios_mavgdaily = train.rolling(window=590, center=False).mean()
std_daily = train.rolling(window=590, center=False).std()
zscore_avg = (ratios_mavghourly - ratios_mavgdaily)/std_daily
plt.figure(figsize=(14,7))
plt.plot(train.index, train.values)
plt.plot(ratios_mavghourly.index, ratios_mavghourly.values)
plt.plot(ratios_mavgdaily.index, ratios_mavgdaily.values)
plt.legend(['Ratio','Hourly Ratio MA', 'Daily Ratio MA'])
plt.ylabel('Ratio')
plt.show()

#Rolling Ratio
zscore_avg.plot(figsize = (14,7))
plt.axhline(0, color='black')
plt.axhline(1.0, color='red', linestyle='--')
plt.axhline(-1.0, color='green', linestyle='--')
plt.legend(['Rolling Ratio z-Score', 'Mean', '+1', '-1'])
plt.show()

#Overall buy/sell plot
ax = train[160:].plot(figsize = (14,7))
buy = train.copy()
sell = train.copy()
buy[zscore_avg>-1] = 0
sell[zscore_avg<1] = 0
buy[160:].plot(ax = ax, color='g', linestyle='None', marker='^', markersize = 3)
sell[160:].plot(ax = ax, color='r', linestyle='None', marker='^', markersize = 3)
x1, x2, y1, y2 = plt.axis()
plt.axis((x1, x2, ratios["close"].min(), ratios["close"].max()))
plt.ylim(0.5,0.7)
plt.legend(['Ratio', 'Buy Signal', 'Sell Signal'])
plt.show()

#Buy/sell plot by company
ax = S1[160:].plot(color='b', figsize = (14,7))
S2[160:].plot(ax = ax, color='m')
buyR = 0*S1.copy()
sellR = 0*S1.copy()
# When you buy the ratio, you buy stock S1 and sell S2
buyR[buy!=0] = S1[buy!=0]
sellR[buy!=0] = S2[buy!=0]
# When you sell the ratio, you sell stock S1 and buy S2
buyR[sell!=0] = S2[sell!=0]
sellR[sell!=0] = S1[sell!=0]
buyR[160:].plot(ax = ax, color='g', linestyle='None', marker='^', markersize = 3)
sellR[160:].plot(ax = ax, color='r', linestyle='None', marker='^', markersize = 3)
x1, x2, y1, y2 = plt.axis()
plt.axis((x1, x2, min(S1["close"].min(), S2["close"].min()), max(S1["close"].max(), S2["close"].max())))
stock1 = str(files[stock_1[highest_cor]])
stock2 = str(files[stock_2[highest_cor]])
plt.xlim(0,8000)
plt.ylim(20,46)
plt.legend([str(stock1[59:]),str(stock2[59:]), 'Buy Signal', 'Sell Signal'])
plt.show()

#basic trading fx
ratios_limit = ratios[:9000]
ma1 = ratios_limit.rolling(window=590, center=False).mean()
ma2 = ratios_limit.rolling(window=60, center=False).mean()
std = ratios_limit.rolling(window=60, center=False).std()

#zscore = (ma1_order - ma2_order) / std_order
zscore_list = zscore_df.values.tolist()
S1 = 0
S2 = 0
money = 0
ratios_list = ratios.values.tolist()
Stock_1_list = Stock_1.values.tolist()
Stock_2_list = Stock_2.values.tolist()
for i in range(0,1000):
    #short S1, long S2
    if (zscore_list[i][0]) > 1:
        money += ((Stock_1_list[i][0]) -(Stock_2_list[i][0])) * float(ratios_list[i][0])
        S1 -= 1
        S2 += 1
    #long S1, short S2
    elif (zscore_list[i][0]) < -1:
        money -= ((Stock_1_list[i][0]) - (Stock_2_list[i][0])) * float(ratios_list[i][0])
        S1 += 1
        S2 -= 1
    # Clear positions
    elif abs((zscore_list[i][0])) < 0.5:
        money += (Stock_1_list[i][0])*float(S1) + ((Stock_2_list[i][0])) * float(S2)
        S1 = 0
        S2 = 0

print(money)


'''
zscore(ratios).plot()
plt.axhline(0, color='black')
plt.axhline(1.0, color='red', linestyle='--')
plt.axhline(-1.0, color='green', linestyle='--')
plt.show()
plt.figure(figsize=(12,6))

ratios_mavg5 = train.rolling(window=5,
                               center=False).mean()
ratios_mavg60 = train.rolling(window=60,
                               center=False).mean()
std_60 = train.rolling(window=60,
                        center=False).std()
zscore_60_5 = (ratios_mavg5 - ratios_mavg60)/std_60
plt.figure(figsize=(15,7))
plt.plot(train.index, train.values)
plt.plot(ratios_mavg5.index, ratios_mavg5.values)
plt.plot(ratios_mavg60.index, ratios_mavg60.values)
plt.legend(['Ratio','5d Ratio MA', '60d Ratio MA'])
plt.ylabel('Ratio')
plt.show()

plt.figure(figsize=(15,7))
zscore_60_5.plot()
plt.axhline(0, color='black')
plt.axhline(1.0, color='red', linestyle='--')
plt.axhline(-1.0, color='green', linestyle='--')
plt.legend(['Rolling Ratio z-Score', 'Mean', '+1', '-1'])
plt.show()

ratios_mavg5 = train.rolling(window=5,
                               center=False).mean()
ratios_mavg60 = train.rolling(window=60,
                               center=False).mean()
std_60 = train.rolling(window=60,
                        center=False).std()
zscore_60_5 = (ratios_mavg5 - ratios_mavg60)/std_60
train[60:].plot()
buy = train.copy()
sell = train.copy()
buy[zscore_60_5>-1] = 0
sell[zscore_60_5<1] = 0
buy[60:].plot(color='g', linestyle='None', marker='^')
sell[60:].plot(color='r', linestyle='None', marker='^')
x1,x2,y1,y2 = plt.axis()
plt.axis((x1,x2,ratios.min(),ratios.max()))
plt.legend(['Ratio', 'Buy Signal', 'Sell Signal'])
plt.show()
plt.figure(figsize=(12,6))
train[160:].plot()
buy = train.copy()
sell = train.copy()

buy[160:].plot(color='g', linestyle='None', marker='^')
sell[160:].plot(color='r', linestyle='None', marker='^')
x1, x2, y1, y2 = plt.axis()
plt.axis((x1, x2, ratios.min(), ratios.max()))
plt.xlim('2013-08-15','2016-07-07')
plt.legend(['Ratio', 'Buy Signal', 'Sell Signal'])
plt.show()

plt.figure(figsize=(12,7))

S1 = dfs[i_index].iloc[:5000]
S2 = dfs[j_index].iloc[5000:]

S1[60:].plot(color='b')
S2[60:].plot(color='c')
buyR = 0*S1.copy()
sellR = 0*S1.copy()

# When you buy the ratio, you buy stock S1 and sell S2
buyR[buy!=0] = S1[buy!=0]
sellR[buy!=0] = S2[buy!=0]

# When you sell the ratio, you sell stock S1 and buy S2
buyR[sell!=0] = S2[sell!=0]
sellR[sell!=0] = S1[sell!=0]

buyR[60:].plot(color='g', linestyle='None', marker='^')
sellR[60:].plot(color='r', linestyle='None', marker='^')
x1, x2, y1, y2 = plt.axis()
plt.axis((x1, x2, min(S1.min(), S2.min()), max(S1.max(), S2.max())))
plt.ylim(25, 105)
plt.xlim('2013-03-22', '2016-07-04')

plt.legend(['AAPL', 'MSFT', 'Buy Signal', 'Sell Signal'])
plt.show()
S1 = dfs[i_index]
S2 = dfs[j_index]
ratios = S1 / S2
ratios.plot()
plt.axhline(ratios.mean())
plt.legend([' Ratio'])
plt.show()
zscore(ratios).plot()
plt.axhline(zscore(ratios).mean())
plt.axhline(1.0, color= 'red')
plt.axhline(-1.0, color= 'green')
plt.show()
plt.figure(figsize=(15,7))
zscore_60_5.plot()
plt.axhline(0, color='black')
plt.axhline(1.0, color='red', linestyle='--')
plt.axhline(-1.0, color='green', linestyle='--')
plt.legend(['Rolling Ratio z-Score', 'Mean', '+1', '-1'])
plt.show()

def PnL(df[i_index], df[j_index]):
    fulldata = pd.concat(df[i_index, df[j_index])
    fulldata_np = np.array(fulldata) # converts our pandas df into numpy array
    nu_shift = np.array(fulldata['nu'].shift(periods=1)) #lag timeseries by 1 trading day
    vol_shift = np.array(fulldata['volatility'].shift(periods=1)) #lag timeseries by 1 trading day
    negvol_shift = np.array(fulldata['negvolatility'].shift(periods=1)) #lag timeseries by 1 trading day
    vec_sig = np.zeros(len(fulldata))
    for i in range(len(fulldata)-1):
        #innovation > volatilty & innovation+1 < volatilty+1
        if (fulldata_np[i,0] > fulldata_np[i,1]): # t1 & t0 SHORT asset when current t changes to nu > volatlity
            vec_sig[i] = -1
        elif(fulldata_np[i,0] < fulldata_np[i,2]): #LONG assets when current t changes to nu < negvol
            vec_sig[i] = 1
        else:
            vec_sig[i] = 0
    # at the current day we enter our position
    vec_sig = pd.DataFrame({'vec_sig':vec_sig})
    return vec_sig
PnL(signals)

print(i)
print(j)
print(files[i])
print(files[j])
print(df1)
print(df2)
print(correlation)
'''