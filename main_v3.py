
import numpy as np
import pandas as pd

##### TODO #########################################
### IMPLEMENT 'getMyPosition' FUNCTION #############
### TO RUN, RUN 'eval.py' ##########################

nInst = 50 # Number of stocks
currentPos = np.zeros(nInst) # Where our positions will be stored
day = 1 # Number of days

prev_slope = np.zeros(nInst) # The previous slope of all 50 stocks
slope_change_threshold = 0.02
lookback = 50  # Number of days to lookback in order to calculate our slope

def getMyPosition(prcSoFar):
    currentPos[:] = -1
    return currentPos


