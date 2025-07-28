# mean(PL): -4.5
# return: -0.00061
# StdDev(PL): 35.28
# annSharpe(PL): -2.01
# totDvolume: 10991649
# Score: -8.03

import numpy as np
import pandas as pd

# STATE
nInst = 50
currentPos = np.zeros(nInst)

# MACD VARIABLES
SHORT_PERIOD = 5
LONG_PERIOD = 10
SIGNAL_PERIOD = 7
previous_macd = []
previous_signals = []
macd_buffer = np.full(nInst, np.nan)
signals_buffer = np.full(nInst, np.nan)

def getMyPosition(prcSoFar):
    global currentPos, nInst
    global SHORT_PERIOD, LONG_PERIOD, SIGNAL_PERIOD
    global previous_macd, previous_signals
    global macd_buffer, signals_buffer

    _, days = prcSoFar.shape

    if days < LONG_PERIOD:
        return np.zeros(50)

    for i in range(nInst):
        todays_price = prcSoFar[i, -1]

        # Required to use the pandas built in EMA calculating method
        prices_i_df = pd.Series(prcSoFar[i, :])

        # Calculate short and long EMAs
        short_ema = prices_i_df.ewm(span=SHORT_PERIOD, adjust=False).mean()
        long_ema = prices_i_df.ewm(span=LONG_PERIOD, adjust=False).mean()

        # Calculate MACD and signal line
        macd = short_ema - long_ema
        macd_buffer[i] = macd.iloc[-1]
        signal = macd.ewm(span=SIGNAL_PERIOD, adjust=False).mean()
        signals_buffer[i] = signal.iloc[-1]

    # Update history
    previous_macd.append(macd_buffer.copy())
    previous_signals.append(signals_buffer.copy())
    macd_history = np.array(previous_macd).T
    signals_history = np.array(previous_signals).T

    if len(previous_macd) > 1:
        prev_macd = np.array(previous_macd[-2])
        prev_signal = np.array(previous_signals[-2])
        curr_macd = macd_buffer
        curr_signal = signals_buffer

        for i in range(nInst):
            if prev_macd[i] < prev_signal[i] and curr_macd[i] > curr_signal[i]:
                currentPos[i] = 10
            elif prev_macd[i] > prev_signal[i] and curr_macd[i] < curr_signal[i]:
                currentPos[i] = -10
            else:
                pass

    return currentPos