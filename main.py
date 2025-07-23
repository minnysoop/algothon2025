# MS is currently working on this, no touchy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

##### TODO #########################################
### IMPLEMENT 'getMyPosition' FUNCTION #############
### TO RUN, RUN 'eval.py' ##########################

nInst = 50
currentPos = np.zeros(nInst)
previousPos = np.zeros(nInst)
day = 1

LOOKBACK = 10

# Estimates
estimate_error = np.ones(nInst)
current_estimate = np.zeros(nInst)
previous_estimate = np.zeros(nInst)

# Measurements
measurement = np.zeros(nInst)
measurement_error = np.ones(nInst)

error = []

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

    # If it's the first day
    if day % 10 == 0 or day == 1:
        # Going to set our initial variables
        current_estimate = today_prices
        estimate_error = np.full(nInst, 10.0)
        measurement = today_prices.copy()
        measurement_error = np.full(nInst, 10.0)
        day += 1
        return currentPos

    if day < LOOKBACK:
        day += 1
        return currentPos

    latest_prices = prcSoFar[:,-LOOKBACK:]

    # Set previous estimates
    previous_estimate = current_estimate.copy()
    # Measure today's stock prices
    measurement = today_prices.copy()
    for i in range(nInst):
        kg = calcKalmanGain(estimate_error[i], measurement_error[i])
        current_estimate[i] = calcNewEstimate(current_estimate[i], kg, measurement[i])
        estimate_error[i] = calcNewError(kg, estimate_error[i])
    error.append(estimate_error[0])


    day += 1
    return currentPos
