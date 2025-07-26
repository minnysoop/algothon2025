# mean(PL): -0.0
# return: -0.00061
# StdDev(PL): 1.81
# annSharpe(PL): -0.27
# totDvolume: 95757
# Score: -0.21


import numpy as np
import pandas as pd

##### TODO #########################################
### IMPLEMENT 'getMyPosition' FUNCTION #############
### TO RUN, RUN 'eval.py' ##########################

nInst = 50 # Number of stocks
currentPos = np.zeros(nInst) # Where our positions will be stored
day = 1 # Number of days

def getMyPosition(prcSoFar):
    global day
    if (day == 1):
        for i in range(50):
            currentPos[i] = 10
    elif(day == 50):
        for i in range(50):
            currentPos[i] = -10
    else:
        for i in range(50):
            currentPos[i] = 0
    day += 1
    return currentPos


