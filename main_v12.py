# mean(PL): 0.1
# return: 0.00056
# StdDev(PL): 4.88
# annSharpe(PL): 0.25
# totDvolume: 207318
# Score: -0.41

import numpy as np
import pandas as pd

# STATE
nInst = 50
currentPos = np.zeros(nInst)
COOLDOWN_DAYS = 5
coolDown = np.full(50, COOLDOWN_DAYS)

# MACD VARIABLES
SHORT_PERIOD = 12
LONG_PERIOD = 26
SIGNAL_PERIOD = 9

def getMyPosition(prcSoFar):
    global currentPos, nInst, coolDown, COOLDOWN_DAYS
    global SHORT_PERIOD, LONG_PERIOD, SIGNAL_PERIOD
    _, days = prcSoFar.shape

    if days < LONG_PERIOD:
        return np.zeros(50)

    normalized_macd = np.full(50, np.nan)
    normalized_signal = np.full(50, np.nan)
    for i in range(nInst):
        todays_price = prcSoFar[i, -1]

        # Required to use the pandas built in EMA calculating method
        prices_i_df = pd.Series(prcSoFar[i, :])

        # Calculate short and long EMAs
        short_ema = prices_i_df.ewm(span=SHORT_PERIOD, adjust=False).mean()
        long_ema = prices_i_df.ewm(span=LONG_PERIOD, adjust=False).mean()

        # Calculate MACD and signal line
        macd = short_ema - long_ema
        signal = macd.ewm(span=SIGNAL_PERIOD, adjust=False).mean()
        delta = macd - signal
        smoothed_delta = delta.rolling(3).mean().iloc[-1]

        normalized_macd[i] = macd.iloc[-1] / todays_price
        normalized_signal[i] = signal.iloc[-1] / todays_price

    for i in range(nInst):
        if coolDown[i] > 0:
            currentPos[i] = 0
            coolDown[i] -= 1
            continue

        macd_val = normalized_macd[i]
        signal_val = normalized_signal[i]

        threshold = 0.01
        if macd_val > signal_val + threshold:
            currentPos[i] = 25
        elif macd_val < signal_val - threshold:
            currentPos[i] = -25
        else:
            currentPos[i] = 0

        coolDown[i] = COOLDOWN_DAYS

    return currentPos