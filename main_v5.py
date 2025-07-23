# MS is currently working on this, no touchy

import numpy as np
import pandas as pd

##### TODO #########################################
### IMPLEMENT 'getMyPosition' FUNCTION #############
### TO RUN, RUN 'eval.py' ##########################


nInst = 50
currentPos = np.zeros(nInst)
day = 1

# Number of days to look back on
LOOKBACK = 10
# Number of stocks
STOCKS = 50

# IN: Array of prices
# OUT: Array with 2 elements [bool, bool]
def calcEndBehavior(arr):
    pass

# INT: Array of prices
# OUT: Array of positions where max (+1) and min (-1) exist
def calcLocalMaxMin(arr):
    pass

# INPUT: Stocks x Days (rows x cols)
def getMyPosition(prcSoFar):
    global day, currentPos
    today_prices = prcSoFar[:,-LOOKBACK:]

    for stock in range(STOCKS):
        pass



    day += 1
    return currentPos



