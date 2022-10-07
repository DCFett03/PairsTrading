# PairsTrading Program
Pairs trading is an investment strategy that identifies 2 companies or funds with similar characteristics and high correlation
whose equity securities are currently trading at a price relationship that is out of their historical trading range. 
This investment strategy will long the undervalued security and short the overvalued security, maintaining market neutrality.
Packages used include pandas, global, numpy, matplotlib, and os packages to create a plotting and trading simulation program. 

To achieve this, the program has several parts. The first of which measured for correlation in a dataset of historical prices, 
and returns the highest correlation as well as the stocks involved.

Then, using z-scores of the data, the program plots various graphs including the relative z-score of the dataset, ratios of the two prices by minutes, 
hours, and days, a rolling ratio, an overall buy/sell plot, and buy/sell plot by stock.

The last part of the program is a trading simulation to try to simulate and predict trading with the previous data and functions.
