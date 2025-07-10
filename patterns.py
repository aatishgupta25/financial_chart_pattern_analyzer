import pandas as pd

def is_doji(df: pd.DataFrame, i: int) -> bool:
    """
    Determines if a candlestick at a given index exhibits the "Doji" pattern.
    A Doji occurs when the open and close prices are nearly identical, indicating market indecision.
    """
    # Out-of-bounds access for 'i' is handled.
    if i >= len(df) or i < 0:
        return False

    row = df.iloc[i]

    # The ticker symbol is dynamically retrieved from the DataFrame's MultiIndex columns.
    try:
        if isinstance(df.columns, pd.MultiIndex):
            ticker_symbol = df.columns[0][1] # The ticker is extracted from the second level of the first column.
        else:
            raise ValueError("DataFrame columns are not a MultiIndex as expected.")

    except (IndexError, ValueError) as e:
        # Errors during ticker symbol determination from DataFrame columns are handled.
        print(f"Error determining ticker symbol from DataFrame columns: {e}")
        # Assuming the structure observed in debug, a re-attempt to extract the ticker is made.
        ticker_symbol = df.columns[0][1]

    # Values are accessed using the MultiIndex tuple.
    open_ = row[('Open', ticker_symbol)]
    close = row[('Close', ticker_symbol)]
    high = row[('High', ticker_symbol)]
    low = row[('Low', ticker_symbol)]

    body = abs(open_ - close)
    range_ = high - low

    # The comparison involves single numerical values, yielding a boolean result.
    return body <= 0.1 * range_

def is_hammer(df: pd.DataFrame, i: int) -> bool:
    """
    Determines if a candlestick at a given index exhibits the "Hammer" pattern.
    A Hammer is a bullish reversal pattern with a small body and a long lower shadow.
    """
    if i >= len(df) or i < 0:
        return False
    row = df.iloc[i]
    if isinstance(df.columns, pd.MultiIndex):
        ticker_symbol = df.columns[0][1]
    else:
        raise ValueError("DataFrame columns are not a MultiIndex as expected in is_hammer.")

    open_ = row[('Open', ticker_symbol)]
    close = row[('Close', ticker_symbol)]
    high = row[('High', ticker_symbol)]
    low = row[('Low', ticker_symbol)]

    body = abs(open_ - close)
    lower_shadow = min(open_, close) - low
    upper_shadow = high - max(open_, close)
    return body < lower_shadow and upper_shadow < body and lower_shadow > 2 * body

def is_bullish_engulfing(df: pd.DataFrame, i: int) -> bool:
    """
    Determines if a candlestick at a given index exhibits the "Bullish Engulfing" pattern.
    This is a two-candle bullish reversal where a large bullish candle completely covers a preceding bearish candle.
    """
    # It is ensured that both the current and previous candles exist.
    if i < 1 or i >= len(df):
        return False

    if isinstance(df.columns, pd.MultiIndex):
        ticker_symbol = df.columns[0][1]
    else:
        raise ValueError("DataFrame columns are not a MultiIndex as expected in is_bullish_engulfing.")

    prev = df.iloc[i - 1]
    curr = df.iloc[i]

    p_open = prev[('Open', ticker_symbol)]
    p_close = prev[('Close', ticker_symbol)]
    c_open = curr[('Open', ticker_symbol)]
    c_close = curr[('Close', ticker_symbol)]

    return (
        p_close < p_open and
        c_close > c_open and
        c_open < p_close and
        c_close > p_open
    )

def is_shooting_star(df: pd.DataFrame, i: int) -> bool:
    """
    Determines if a candlestick at a given index exhibits the "Shooting Star" pattern.
    This is a bearish reversal pattern with a small body and a long upper shadow.
    """
    if i >= len(df) or i < 0:
        return False
    row = df.iloc[i]
    if isinstance(df.columns, pd.MultiIndex):
        ticker_symbol = df.columns[0][1]
    else:
        raise ValueError("DataFrame columns are not a MultiIndex as expected in is_shooting_star.")

    open_ = row[('Open', ticker_symbol)]
    close = row[('Close', ticker_symbol)]
    high = row[('High', ticker_symbol)]
    low = row[('Low', ticker_symbol)]

    body = abs(open_ - close)
    upper_shadow = high - max(open_, close)
    lower_shadow = min(open_, close) - low
    return body < upper_shadow and lower_shadow < body and upper_shadow > 2 * body

def detect_patterns(df: pd.DataFrame, i: int) -> str:
    """
    Detects known candlestick patterns at a specific index within the DataFrame.
    """
    if is_doji(df, i):
        return "doji"
    elif is_hammer(df, i):
        return "hammer"
    elif is_bullish_engulfing(df, i):
        return "bullish engulfing"
    elif is_shooting_star(df, i):
        return "shooting star"
    else:
        return "no pattern"