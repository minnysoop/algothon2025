import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# PROVIDED VARIABLES
nInst = 50  # Number of available instruments
currentPos = np.zeros(nInst)  # Positions per day
day = 1  # Number of days

# EXTRA VARIABLES
entry_prices = np.full(nInst, np.nan)

# PROGRAM ADJUSTERS
LOOKBACK = 10
STOP_LOSS = 0.03
STOP_GAIN = 0.01

# VARIABLES FOR EMA
previous_emas = [None]*nInst
ema_history = []
SMOOTHING_FACTOR = 2/(LOOKBACK+1)

# VARIABLES FOR FIRST AND SECOND DERIVATIVES
current_first_deriv = np.zeros(nInst)
current_second_deriv = np.zeros(nInst)
first_deriv_history = []
second_deriv_history = []

# Calculates EMA for a single stock
def calc_ema(prices, prev_ema=None):
    global SMOOTHING_FACTOR
    if prev_ema is None:
        return np.mean(prices)
    else:
        return SMOOTHING_FACTOR * prices[-1] + (1 - SMOOTHING_FACTOR) * prev_ema

def getMyPosition(prcSoFar):
    # prcSoFar holds all prices up for all nInst instruments to the current day
    global day, currentPos
    global LOOKBACK
    global ema_history, previous_emas

    # If there isn't enough data, don't trade
    if day < LOOKBACK:
        day += 1
        return currentPos

    # CALCULATE NEW EMA
    for i in range(nInst):
        # Get latest prices from LOOKBACK period
        latest_prices = prcSoFar[i, -LOOKBACK:]
        current_ema = calc_ema(latest_prices, previous_emas[i])
        previous_emas[i] = current_ema
    # Update EMA history
    ema_history.append(previous_emas.copy())
    ema_history_matrix = np.array(ema_history).T

    # Need to have calculated 3 EMAs to find derivative
    if len(ema_history) >= 3:
        # CALCULATE DERIVATIVES
        for i in range(nInst):
            first_derivative = ema_history_matrix[i, -1] - ema_history_matrix[i, -2]
            current_first_deriv[i] = first_derivative
            second_derivative = ema_history_matrix[i, -1] - 2 * ema_history_matrix[i, -2] + ema_history_matrix[i, -3]
            current_second_deriv[i] = second_derivative
        # Update derivatives history
        first_deriv_history.append(current_first_deriv.copy())
        second_deriv_history.append(current_second_deriv.copy())
        first_derivative_matrix = np.array(first_deriv_history).T
        second_derivative_matrix = np.array(second_deriv_history).T

        ## SIGNAL LOGIC START
        for i in range(nInst):
            current_price_stock_i = prcSoFar[i, -1]
            current_ema_stock_i = ema_history_matrix[i, -1]
            current_first_deriv_stock_i = first_derivative_matrix[i, -1]
            current_second_deriv_stock_i = second_derivative_matrix[i, -1]

            ## STOP LOSS / STOP GAIN LOGIC
            if not np.isnan(entry_prices[i]):
                price_change = (current_price_stock_i - entry_prices[i]) / entry_prices[i]
                if price_change >= STOP_GAIN or price_change <= -STOP_LOSS:
                    currentPos[i] = -10
                    entry_prices[i] = np.nan
                    continue

            if abs(current_first_deriv_stock_i) <= 0.002:
                if current_second_deriv_stock_i >= 0.0 and np.isnan(entry_prices[i]):
                    currentPos[i] = 10
                    entry_prices[i] = current_price_stock_i
                elif current_second_deriv_stock_i < 0.0 and not np.isnan(entry_prices[i]):
                    currentPos[i] = -10
                    entry_prices[i] = np.nan
        ## SIGNAL LOGIC ENDS

        # if day == 1499:
        #     stock_i = 0
        #     # Grab price series aligned with EMA history length
        #     ema_series = ema_history_matrix[stock_i, LOOKBACK:]
        #     prices = prcSoFar[stock_i, -len(ema_series):]
        #
        #     # Convert derivative histories to arrays
        #     first_derivative_matrix = np.array(first_deriv_history).T
        #     second_derivative_matrix = np.array(second_deriv_history).T
        #
        #     # Calculate dynamic padding length for derivatives to align with ema_series
        #     num_padding = abs(len(ema_series) - len(first_derivative_matrix[stock_i]))
        #     first_derivative = np.concatenate((np.zeros(num_padding), first_derivative_matrix[stock_i]))
        #     second_derivative = np.concatenate((np.zeros(num_padding), second_derivative_matrix[stock_i]))
        #
        #     x_vals = np.arange(len(prices))
        #
        #     fig = go.Figure()
        #
        #     fig.add_trace(go.Scatter(x=x_vals, y=prices, mode='lines', name='Raw Prices', line=dict(color='blue')))
        #     fig.add_trace(go.Scatter(x=x_vals, y=ema_series, mode='lines', name='EMA', line=dict(color='orange')))
        #     fig.add_trace(go.Scatter(x=x_vals, y=first_derivative, mode='lines', name='First Derivative',
        #                              line=dict(color='green', dash='dash')))
        #     fig.add_trace(go.Scatter(x=x_vals, y=second_derivative, mode='lines', name='Second Derivative',
        #                              line=dict(color='red', dash='dash')))
        #
        #     fig.update_layout(
        #         title=f"Stock {stock_i} Price and Derivatives at Day {day}",
        #         xaxis_title="Days",
        #         yaxis_title="Value",
        #         legend=dict(x=0, y=1),
        #         template="plotly_white",
        #         hovermode="x unified",
        #     )
        #     fig.show()

    day += 1
    return currentPos



