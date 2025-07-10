import yfinance as yf
import pandas as pd
import os
import mplfinance as mpf
from patterns import detect_patterns


# List of stock tickers for data retrieval.
TICKERS = ['AAPL', 'MSFT', 'GOOG', 'TSLA', 'AMZN']
# Base directory for saving generated data.
SAVE_DIR = 'data'
# Subdirectory for saving chart images.
CHART_DIR = os.path.join(SAVE_DIR, 'charts')
# Subdirectory for saving pattern labels.
LABEL_DIR = os.path.join(SAVE_DIR, 'labels')

# Directories are ensured to exist.
os.makedirs(CHART_DIR, exist_ok=True)
os.makedirs(LABEL_DIR, exist_ok=True)

def save_chart(df: pd.DataFrame, index: int, pattern: str):
    """
    Saves a candlestick chart image and its corresponding pattern label.

    Parameters:
        df (pd.DataFrame): DataFrame containing the OHLCV data for the chart window.
        index (int): A unique index used for file naming.
        pattern (str): The detected pattern label.
    """
    fname = f"img_{index:05d}"
    df_plot = df.copy()

    # The ticker symbol is determined from the DataFrame's MultiIndex columns.
    # This assumes the DataFrame (representing a chart window) retains the MultiIndex structure
    # from the yfinance download.
    ticker_symbol = None
    if isinstance(df_plot.columns, pd.MultiIndex):
        # The ticker is extracted from the second level of the first available column.
        for col_tuple in df_plot.columns:
            if isinstance(col_tuple, tuple) and len(col_tuple) > 1:
                ticker_symbol = col_tuple[1]
                break
        if ticker_symbol is None:
            raise ValueError("Ticker symbol could not be determined from MultiIndex columns in save_chart.")
    else:
        # An error is raised if the DataFrame columns are not a MultiIndex as expected.
        raise ValueError("DataFrame columns are not a MultiIndex as expected in save_chart.")

    # Columns are renamed to simple names for mplfinance, which expects 'Open', 'High', etc.
    # A new DataFrame is created with single-level column names using selected columns.
    new_df_for_mpf = pd.DataFrame(index=df_plot.index)
    new_df_for_mpf['Open'] = df_plot[('Open', ticker_symbol)]
    new_df_for_mpf['High'] = df_plot[('High', ticker_symbol)]
    new_df_for_mpf['Low'] = df_plot[('Low', ticker_symbol)]
    new_df_for_mpf['Close'] = df_plot[('Close', ticker_symbol)]
    # Volume column is included if present in the original DataFrame.
    if ('Volume', ticker_symbol) in df_plot.columns:
        new_df_for_mpf['Volume'] = df_plot[('Volume', ticker_symbol)]

    # Data types for relevant columns are ensured to be float.
    for col in ['Open', 'High', 'Low', 'Close']:
        new_df_for_mpf[col] = new_df_for_mpf[col].astype(float)
    if 'Volume' in new_df_for_mpf.columns:
         new_df_for_mpf['Volume'] = new_df_for_mpf['Volume'].astype(float)

    # The DataFrame index is converted to datetime, as required by mplfinance.
    new_df_for_mpf.index = pd.to_datetime(new_df_for_mpf.index)

    # The chart is plotted and saved as a PNG image.
    mpf.plot(new_df_for_mpf, type='candle', style='charles',
             savefig=os.path.join(CHART_DIR, f"{fname}.png"))

    # The pattern label is saved to a text file.
    with open(os.path.join(LABEL_DIR, f"{fname}.txt"), 'w') as f:
        f.write(pattern)

def generate_dataset(num_per_ticker=500):
    """
    Generates a dataset of candlestick charts and labels by fetching stock data.

    Parameters:
        num_per_ticker (int): The maximum number of charts to save per ticker.
    """
    count = 0
    for ticker in TICKERS:
        print(f"Fetching {ticker}")
        # Financial data is downloaded, which results in a DataFrame with MultiIndex columns.
        df = yf.download(ticker, period='1y', interval='1d')
        # Rows with any missing values are removed.
        df.dropna(inplace=True)

        for i in range(10, len(df)):
            # A 10-day window of historical data is extracted.
            window = df.iloc[i-10:i]
            # Candlestick patterns are detected using the full DataFrame and current index.
            label = detect_patterns(df, i)
            if label != "no pattern":
                save_chart(window, count, label)
                count += 1
                # Data generation is stopped once the target number of charts is reached.
                if count >= len(TICKERS) * num_per_ticker:
                    print(f"Saved {count} charts")
                    return

# The dataset generation process is initiated when the script is run directly.
if __name__ == "__main__":
    generate_dataset()