import numpy as np

def getMyPosition(prices):
    """
    Calculate desired positions for 50 instruments based on price data.
    
    Parameters:
    prices (numpy.ndarray): Array of shape (50, nt) where each row is an instrument
                           and each column is a day's price.
    
    Returns:
    numpy.ndarray: Vector of 50 integers representing desired positions (shares).
    """
    nInst, nt = prices.shape
    positions = np.zeros(nInst, dtype=int)
    
    # Parameters
    lookback_long = 30   # Longer-term SMA for stronger trend detection
    lookback_short = 10  # Longer short-term SMA for smoother signals
    lookback_rsi = 14    # RSI period
    rsi_overbought = 80  # Tight overbought threshold
    rsi_oversold = 20    # Tight oversold threshold
    position_limit = 10000  # $10,000 limit per stock
    volatility_threshold = 0.07  # Lowered threshold for contrarian twist
    
    for inst in range(nInst):
        # Get price series for the instrument
        price_series = prices[inst, :]
        
        # Skip if not enough data
        if nt < lookback_long:
            continue
            
        # Calculate SMAs
        sma_long = np.mean(price_series[-lookback_long:])
        sma_short = np.mean(price_series[-lookback_short:])
        
        # Calculate RSI
        if nt >= lookback_rsi:
            deltas = np.diff(price_series[-lookback_rsi-1:])
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            avg_gain = np.mean(gains) if len(gains) > 0 else 0
            avg_loss = np.mean(losses) if len(losses) > 0 else 1e-10
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
        else:
            rsi = 50  # Neutral RSI if insufficient data
            
        # Calculate volatility (standard deviation of returns)
        if nt >= lookback_long:
            returns = np.diff(price_series[-lookback_long:]) / price_series[-lookback_long:-1]
            volatility = np.std(returns) if len(returns) > 0 else 1e-10
        else:
            volatility = 1e-10
            
        # Current price
        current_price = price_series[-1]
        
        # Generate trading signal (enhanced momentum with adjusted contrarian twist)
        signal = 0
        if sma_short > sma_long and rsi < rsi_overbought:
            if volatility > volatility_threshold:
                signal = -1  # Contrarian sell in moderate to high volatility
            else:
                signal = 1   # Momentum buy
        elif sma_short < sma_long and rsi > rsi_oversold:
            if volatility > volatility_threshold:
                signal = 1   # Contrarian buy in moderate to high volatility
            else:
                signal = -1  # Momentum sell
            
        # Volatility-based position sizing
        base_position = position_limit / current_price  # Max shares based on $10k limit
        volatility_adjustment = max(0.5, min(0.9, 1 / (1 + 2 * volatility)))  # Conservative scaling
        target_shares = base_position * volatility_adjustment
        
        # Apply signal and round to integer
        positions[inst] = int(target_shares * signal)
        
        # Ensure position is within $10,000 limit
        dollar_position = abs(positions[inst] * current_price)
        if dollar_position > position_limit:
            positions[inst] = int(np.sign(positions[inst]) * (position_limit / current_price))
    
    return positions