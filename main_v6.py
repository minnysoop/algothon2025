# IGNORE THIS

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go

##### TODO #########################################
### IMPLEMENT 'getMyPosition' FUNCTION #############
### TO RUN, RUN 'eval.py' ##########################

nInst = 50  # Number of stocks
currentPos = np.zeros(nInst)  # Where our positions will be stored
previousPos = np.zeros(nInst) # The most previous position
day = 1  # Number of days

# Constants
LOOKBACK = 20 # Number of days to lookback on
STOP_LOSS = 0.01 # Exit when you get a 1% decrease
STOP_GAIN = 0.01 # Exit when you get a 1% increase
entry_prices = np.full(nInst, np.nan) # Represents the price at which you entered the stock

# Variables for exponential moving averages (EMA)
SMOOTHING_FACTOR = 2/(LOOKBACK+1) # Smoothing factor for EMA
previous_ema = np.zeros(50) # Stores the "most previous" EMAs for all 50 instruments
ema_history = [] # Stores all previous EMAs for all 50 stocks

# Variables for bollinger bands
bollinger_upper = []
bollinger_lower = []
previous_upper = np.zeros(50)
previous_lower = np.zeros(50)

# Variables for derivatives
FIRST_DERIV_THRESHOLD = 0.05
SECOND_DERIV_THRESHOLD = 0.1

# Variables for Kalman Filter
estimate_error = np.ones(nInst)
current_estimate = np.zeros(nInst)
previous_estimate = np.zeros(nInst)
measurement = np.zeros(nInst)
measurement_error = np.ones(nInst)
kalman_estimate_history = []

# Calculates the EMA for one stock
def ema(prices, prev_ema=None):
    global SMOOTHING_FACTOR
    if prev_ema is None:
        return np.mean(prices)
    else:
        return SMOOTHING_FACTOR * prices[-1] + (1 - SMOOTHING_FACTOR) * prev_ema

# Calculates the upper and lower bollinger bands
def bollinger_bands(prices, k=2.0):
    std = np.std(prices)
    mid = ema(prices)
    upper = mid + k * std
    lower = mid - k * std
    return lower, upper

def calcKalmanGain(error_est, error_mea):
    return error_est / (error_est + error_mea)

def calcNewEstimate(prev_est, kg, mea):
    return prev_est + kg * (mea - prev_est)

def calcNewError(kg, prev_error_estimate):
    return (1 - kg) * prev_error_estimate

# Plots stuff for us
def plot(prices, ema_series, upper_band_series, lower_band_series, kalman_series):
    fig = go.Figure()

    fig.add_trace(go.Scatter(y=prices, mode='lines', name='Raw Prices', line=dict(color='blue')))
    fig.add_trace(go.Scatter(y=ema_series, mode='lines', name='EMA', line=dict(color='orange')))
    fig.add_trace(go.Scatter(y=kalman_series, mode='lines', name='Kalman Estimate', line=dict(color='purple')))
    fig.add_trace(
        go.Scatter(y=upper_band_series, mode='lines', name='Bollinger Upper', line=dict(color='green', dash='dash')))
    fig.add_trace(
        go.Scatter(y=lower_band_series, mode='lines', name='Bollinger Lower', line=dict(color='red', dash='dash')))

    fig.update_layout(
        title=f"Stock {0} - Price, EMA, Kalman, Bollinger",
        xaxis_title="Days",
        yaxis_title="Price",
        legend=dict(x=0, y=1),
        template="plotly_white",
        hovermode="x unified",
    )

    fig.show()

def getMyPosition(prcSoFar):
    global day, currentPos, previousPos
    global previous_ema, ema_history
    global bollinger_lower, bollinger_upper, previous_upper, previous_lower
    global STOP_LOSS, STOP_GAIN, entry_prices
    global FIRST_DERIV_THRESHOLD, SECOND_DERIV_THRESHOLD
    global current_estimate, estimate_error, previous_estimate
    global measurement, measurement_error

    today_prices = prcSoFar[:, -1]

    # If it's the first day
    if day == 1:
        # Going to set our initial variables
        current_estimate = today_prices
        estimate_error = np.full(nInst, 10.0)
        measurement = today_prices.copy()
        measurement_error = np.full(nInst, 10.0)
        currentPos = np.zeros(nInst)
        day += 1
        return currentPos

    # If the number of days is less than the LOOKBACK period
    if prcSoFar.shape[1] < LOOKBACK:
        # Don't do anything
        day += 1
        return currentPos

    previous_estimate = current_estimate.copy()
    # Measure today's stock prices
    measurement = today_prices.copy()
    # For every stock
    for i in range(nInst):
        # Get latest prices for ith stock
        latest_prices_for_stock_i = prcSoFar[i, -LOOKBACK:]

        # Calculate EMA for the ith stock
        new_ema = ema(latest_prices_for_stock_i, previous_ema[i])
        previous_ema[i] = new_ema

        # Compute Bollinger Bands
        new_lower, new_upper = bollinger_bands(latest_prices_for_stock_i, k=1.75)
        previous_upper[i] = new_upper
        previous_lower[i] = new_lower

        # Kalman Filter
        kg = calcKalmanGain(estimate_error[i], measurement_error[i])
        # Calculate new estimates
        current_estimate[i] = calcNewEstimate(current_estimate[i], kg, measurement[i])
        estimate_error[i] = calcNewError(kg, estimate_error[i])

        signal = current_estimate[i] - today_prices[i]
        currentPos[i] = np.clip(signal, -100, 100) * 100

    # Update EMA history and Bollinger Bands history with the new values
    ema_history.append(previous_ema.copy())
    bollinger_upper.append(previous_upper.copy())
    bollinger_lower.append(previous_lower.copy())
    kalman_estimate_history.append(current_estimate.copy())

    # Preprocessing step to be able to access ema history and bollinger bands for ith stock like [i, :]
    ema_history_matrix = np.array(ema_history).T
    upper_band_matrix = np.array(bollinger_upper).T
    lower_band_matrix = np.array(bollinger_lower).T
    kalman_matrix = np.array(kalman_estimate_history).T

    # TMP DELETE LATER
    for i in range(nInst):
        latest_price = prcSoFar[i, -1]
        cur_ema = ema_history_matrix[i, -1]
        upper = upper_band_matrix[i, -1]
        lower = lower_band_matrix[i, -1]

        if len(ema_history) < 3 or cur_ema < lower:
            continue

        if not np.isnan(entry_prices[i]):
            price_change = (latest_price - entry_prices[i]) / entry_prices[i]
            if price_change >= STOP_GAIN or price_change <= -STOP_LOSS:
                currentPos[i] = 0
                entry_prices[i] = np.nan
                continue

        # Discrete first and second derivative, basically a linear approximation
        first_derivative = cur_ema - ema_history_matrix[i, -2]
        second_derivative = cur_ema - 2*ema_history_matrix[i, -2] + ema_history_matrix[i, -3]

        if abs(latest_price - upper) <= 0.01:
            # Go short
            currentPos[i] = -25
        elif abs(latest_price - lower) <= 0.01 and not np.isnan(entry_prices[i]):
            # Go long
            currentPos[i] = 50
            entry_prices[i] = latest_price
        # else:
        #     currentPos[i] = 0

        # if abs(latest_price - upper) <= 0.01:
        #     currentPos[i] = -50


    if day == 1499:
        stock_i = 0
        prices = prcSoFar[stock_i, :]
        ema_series = ema_history_matrix[stock_i, :]
        upper_band_series = upper_band_matrix[stock_i, :]
        lower_band_series = lower_band_matrix[stock_i, :]
        kalman_series = kalman_matrix[stock_i, :]
        plot(prices, ema_series, upper_band_series, lower_band_series, kalman_series)

    # Updating previous position
    previousPos = currentPos.copy()
    day += 1
    return currentPos



