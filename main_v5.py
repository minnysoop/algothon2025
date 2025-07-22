# MS is currently working on this, no touchy

import numpy as np
import pandas as pd

##### TODO #########################################
### IMPLEMENT 'getMyPosition' FUNCTION #############
### TO RUN, RUN 'eval.py' ##########################

nInst = 50  # Number of stocks
currentPos = np.zeros(nInst)  # Where our positions will be stored
day = 1  # Number of days

# Estimates
estimate_error = np.ones(nInst)
current_estimate = np.zeros(nInst)
previous_estimate = np.zeros(nInst)

# Measurements
measurement = np.zeros(nInst)
measurement_error = np.ones(nInst)

def calcKalmanGain(error_est, error_mea):
    return error_est / (error_est + error_mea)

def calcNewEstimate(prev_est, kg, mea):
    return prev_est + kg * (mea - prev_est)

def calcNewError(kg, prev_error_estimate):
    return (1 - kg) * prev_error_estimate

def getMyPosition(prcSoFar):
    global day, currentPos
    global current_estimate, estimate_error, previous_estimate
    global measurement, measurement_error

    today_prices = prcSoFar[:, -1]

    if day % 10 == 0 or day == 1:
        current_estimate = today_prices.copy()
        estimate_error = np.full(nInst, 10.0)
        measurement = today_prices.copy()
        measurement_error = np.full(nInst, 10.0)
        currentPos = np.zeros(nInst)
        day += 1
        return currentPos

    previous_estimate = current_estimate.copy()

    measurement = today_prices.copy()
    for i in range(nInst):
        kg = calcKalmanGain(estimate_error[i], measurement_error[i])
        current_estimate[i] = calcNewEstimate(previous_estimate[i], kg, measurement[i])
        estimate_error[i] = calcNewError(kg, estimate_error[i])


    day += 1
    return currentPos



