import numpy as np

# PROGRAM VARIABLES
nInst = 50
currentPos = np.zeros(nInst)
day = 1

# EMA VARIABLES
previous_emas = [None]*nInst
ema_history = []
LOOKBACK = 10
SMOOTHING_FACTOR = 2/(LOOKBACK+1)

# DERIVATIVE VARIABLES
current_first_deriv = np.zeros(nInst)
current_second_deriv = np.zeros(nInst)
first_deriv_history = []
second_deriv_history = []
WINDOW = 20

# Calculates the exponential moving average for given prices
def calc_ema(prices, prev_ema=None):
    global SMOOTHING_FACTOR
    if prev_ema is None:
        return np.mean(prices)
    else:
        return SMOOTHING_FACTOR * prices[-1] + (1 - SMOOTHING_FACTOR) * prev_ema

# Determines whether the list is increasing
def is_increasing(l):
    for i in range(1, len(l)):
        if l[i] - l[i-1] < 0.001:
            return False
    return True

# Finds the number of inflection points given previous second derivatives
def inflection_points(ddx):
    count = 0
    for i in range(1, len(ddx)):
        prev = ddx[i - 1]
        curr = ddx[i]
        if np.sign(prev) != np.sign(curr) and abs(prev - curr) > 0.001:
            count += 1
    return count

def getMyPosition(prcSoFar):
    # GLOBALS
    global day, currentPos
    global LOOKBACK, WINDOW
    global ema_history, previous_emas
    global current_first_deriv, current_second_deriv, first_deriv_history, second_deriv_history

    # If there isn't sufficient data
    if day < LOOKBACK:
        day += 1
        return np.zeros(50) # Don't do anything

    # For every stock
    for i in range(nInst):
        # Get LOOKBACK number of latest prices
        latest_prices = prcSoFar[i, -LOOKBACK:]
        # Calculate current EMA
        current_ema = calc_ema(latest_prices, previous_emas[i])
        # Update the EMA buffer
        previous_emas[i] = current_ema
    # Update EMA history
    ema_history.append(previous_emas.copy())
    ema_history_matrix = np.array(ema_history).T

    # If there is sufficient EMAs to calculate the derivaitives
    if len(ema_history) >= 3:
        # For every stock
        for i in range(nInst):
            # Calculate first and second derivatives (basically linear approximations)
            first_derivative = ema_history_matrix[i, -1] - ema_history_matrix[i, -2]
            current_first_deriv[i] = first_derivative
            second_derivative = ema_history_matrix[i, -1] - 2 * ema_history_matrix[i, -2] + ema_history_matrix[i, -3]
            current_second_deriv[i] = second_derivative
        # Update derivatives history
        first_deriv_history.append(current_first_deriv.copy())
        second_deriv_history.append(current_second_deriv.copy())
        first_derivative_matrix = np.array(first_deriv_history).T
        second_derivative_matrix = np.array(second_deriv_history).T

        # DECISION LOGIC START
        for i in range(nInst):
            current_price_stock_i = prcSoFar[i, -1]
            current_first_deriv_stock_i = first_derivative_matrix[i, -1]
            current_second_deriv_stock_i = second_derivative_matrix[i, -1]
            past_second_deriv_stock_i = second_derivative_matrix[i, -WINDOW:]
            past_first_deriv_stock_i = first_derivative_matrix[i, -WINDOW:]

            # Handle volatility with the second derivative
            inflpoints = inflection_points(past_second_deriv_stock_i)
            if inflpoints > WINDOW/2:
                currentPos[i] = 0
                continue

            target_dollar_per_stock = 5000 # How much money we can spend per stock
            # If there is a local extrema
            if abs(current_first_deriv_stock_i) <= 0.001:
                # It's increasing (concave upwards)
                if is_increasing(past_second_deriv_stock_i) and current_second_deriv_stock_i > 0:
                    currentPos[i] = target_dollar_per_stock/current_price_stock_i # Adjusted position based on target dollar_per_stock
                # If it's decreasing (concave downwards
                elif not is_increasing(past_second_deriv_stock_i) and current_second_deriv_stock_i < 0:
                    currentPos[i] = -target_dollar_per_stock/current_price_stock_i
            # If it's losing momentum up to the extrema and currentPosition is long
            elif currentPos[i] > 0 and not is_increasing(past_first_deriv_stock_i):
                currentPos[i] = 0 # Exit
                continue
            # If it's gaining momentum and currentPosition is short
            elif currentPos[i] < 0 and is_increasing(past_first_deriv_stock_i):
                currentPos[i] = 0 # Exit
                continue

        # DECISION LOGIC END

    day += 1 # Update day
    return currentPos



