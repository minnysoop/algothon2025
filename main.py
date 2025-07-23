# Kalman Filter template

import numpy as np
import pandas as pd

##### TODO #########################################
### IMPLEMENT 'getMyPosition' FUNCTION #############
### TO RUN, RUN 'eval.py' ##########################

nInst = 50  # Number of stocks
currentPos = np.zeros(nInst)  # Where our positions will be stored
previousPos = np.zeros(nInst)
day = 1  # Number of days

LOOKBACK = 30

def getMyPosition(prcSoFar):
    global day, currentPos

    current_prices = prcSoFar[:, -1]
    latest_prices = prcSoFar[:, -LOOKBACK:]

    for i in range(50):
        median = np.median(latest_prices[i])
        if median < current_prices[i]:
            currentPos[i] = -10
        else:
            currentPos[i] = 10

    day += 1
    return currentPos



