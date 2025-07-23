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

# Number of days to look back on
LOOKBACK = 20
# Number of stocks
STOCKS = 50

def plot_kalman_vs_price(prices, stock_id=0):
    smoothed = kalman_filter(prices)
    ma, upper, lower = bollinger_bands(prices)

    plt.figure(figsize=(12, 5))
    # plt.plot(prices, label="Raw Prices", color="gray", alpha=0.4)
    plt.plot(smoothed, label="Kalman Filter", color="dodgerblue", linewidth=2)
    plt.plot(ma, label="Bollinger MA", color="orange", linestyle='--')
    plt.plot(upper, label="Bollinger Upper", color="green", linestyle='--', alpha=0.7)
    plt.plot(lower, label="Bollinger Lower", color="red", linestyle='--', alpha=0.7)

    plt.fill_between(range(len(prices)), lower, upper, color='lightgray', alpha=0.2)

    plt.title(f"Stock {stock_id} - Kalman Filter & Bollinger Bands")
    plt.xlabel("Day")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def bollinger_bands(prices, window=20, num_std=2):

    rolling_mean = pd.Series(prices).rolling(window).mean()
    rolling_std = pd.Series(prices).rolling(window).std()

    upper_band = rolling_mean + (num_std * rolling_std)
    lower_band = rolling_mean - (num_std * rolling_std)

    return rolling_mean.to_numpy(), upper_band.to_numpy(), lower_band.to_numpy()


def kalman_filter(prices, process_var=1e-5, measurement_var=0.01):
    n = len(prices)
    x = np.zeros(n)
    P = np.zeros(n)
    x[0] = prices[0]
    P[0] = 1.0

    Q = process_var
    R = measurement_var

    for t in range(1, n):
        x_pred = x[t-1]
        P_pred = P[t-1] + Q

        K = P_pred / (P_pred + R)
        x[t] = x_pred + K * (prices[t] - x_pred)
        P[t] = (1 - K) * P_pred

    return x

def getMyPosition(prcSoFar):
    global day, currentPos, previousPos
    global STOCKS, LOOKBACK

    if day < LOOKBACK:
        day += 1
        return currentPos

    latest_prices = prcSoFar[:, -LOOKBACK:]
    previousPos = currentPos.copy()
    currentPos = np.zeros(STOCKS)

    for i in range(STOCKS):
        raw_prices = latest_prices[i]

        # Step 1: Apply Kalman filter to raw prices
        smoothed_prices = kalman_filter(raw_prices)

        # Step 2: Compute Bollinger Bands on smoothed prices
        ma, upper, lower = bollinger_bands(smoothed_prices)

        # Step 3: Use last price (most recent day)
        current_price = smoothed_prices[-1]

        if np.isnan(lower[-1]) or np.isnan(upper[-1]):
            continue  # Not enough data for BB calc

        # Step 4: Signal logic
        if current_price < lower[-1]:
            currentPos[i] = 1  # BUY signal (oversold)
        elif current_price > upper[-1]:
            currentPos[i] = -1  # SELL signal (overbought)
        else:
            currentPos[i] = 0  # FLAT


    day += 1
    return currentPos