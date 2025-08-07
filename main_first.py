
import numpy as np

##### TODO #########################################
### IMPLEMENT 'getMyPosition' FUNCTION #############
### TO RUN, RUN 'eval.py' ##########################

nInst = 50 # 50 Stocks from 50 different companies
currentPos = np.zeros(nInst)

# Represents the direction in which the prices are going towards
direction = np.zeros(nInst)

# Represents, for each stock, if it went up or down
current_ups = np.zeros(direction.shape, dtype=bool)

threshold = 50
streak_matrix = np.empty((threshold, 50), dtype=bool)

# Days passed
day = 1

def getMyPosition(prcSoFar):
    global direction, currentPos, day, current_ups

    current_day_prices = prcSoFar[:, -1]

    if (direction == 0).all():
        direction = prcSoFar[:, -1]
        day += 1
        return currentPos

    current_ups = current_day_prices > direction
    direction = current_day_prices.copy()

    if day <= threshold:
        streak_matrix[day - 1] = current_ups
    else:
        streak_matrix[:-1] = streak_matrix[1:]
        streak_matrix[-1] = current_ups

    # Time to sell/buy stuff
    transposed_streak_matrix = streak_matrix.transpose()

    for i in range(len(currentPos)):
        last_couple_days = transposed_streak_matrix[i]
        momentum = np.sum(last_couple_days)

        max_buy = 10
        max_sell = -10

        currentPos[i] = (momentum / threshold) * (max_buy - max_sell) + max_sell

    day += 1
    return currentPos


