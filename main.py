
import numpy as np
import pandas as pd

##### TODO #########################################
### IMPLEMENT 'getMyPosition' FUNCTION #############
### TO RUN, RUN 'eval.py' ##########################

nInst = 50 # Number of stocks
currentPos = np.zeros(nInst) # Where our positions will be stored
day = 1 # Number of days

prev_slope = np.zeros(nInst) # The previous slope of all 50 stocks
slope_change_threshold = 0.01
lookback = 30  # Number of days to lookback in order to calculate our slope

def getMyPosition(prcSoFar):
    global currentPos, day, prev_slope

    # If we don't have enough days to lookback on to compute a slope
    if prcSoFar.shape[0] < lookback:
        # Don't trade
        return np.zeros(nInst)

    # Computing our positions
    for i in range(nInst):
        # 'lookback' number of prices for the ith stock
        prices = prcSoFar[-lookback:, i]

        # Volatility measurement based on coefficient of variation
        # "How much stock fluctuates relative to its average price"
        volatility = np.std(prices) / np.mean(prices)
        # If for the pass lookback days, it's not that volatile
        if volatility < 0.1:
            currentPos[i] = 0
            continue # Probably best not to buy because not enough movement, best to just hold

        # Calculating EMAs (Exponential Moving Averages),
        # which really is a type of moving average that gives more weight to the most recent data than the old ones
        prices_series = pd.Series(prices)
        ema = prices_series.ewm(span=lookback, adjust=False).mean()
        slope = ema.iloc[-1] - ema.iloc[-2]

        if slope > slope_change_threshold and prev_slope[i] <= slope_change_threshold:
            currentPos[i] = 1
        elif slope < -slope_change_threshold and prev_slope[i] >= -slope_change_threshold:
            currentPos[i] = -1
        else:
            currentPos[i] = 0

        prev_slope[i] = slope

    day += 1
    return currentPos


