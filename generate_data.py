import yfinance as yf
import pandas as pd
import os
import mplfinance as mpf
from patterns import detect_patterns

TICKERS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'BRK.B', 'UNH', 'TSLA', 'LLY',
    'JPM', 'JNJ', 'V', 'XOM', 'PG', 'AVGO', 'MA', 'HD', 'MRK', 'PEP',
    'COST', 'ABBV', 'ADBE', 'CVX', 'KO', 'WMT', 'BAC', 'MCD', 'TMO', 'PFE',
    'DIS', 'CRM', 'NFLX', 'ACN', 'ABT', 'INTC', 'CMCSA', 'VZ', 'ORCL', 'NKE',
    'QCOM', 'DHR', 'TXN', 'NEE', 'LIN', 'WFC', 'IBM', 'AMGN', 'MDT', 'UPS'
    ]
# TICKERS = ['AAPL', 'MSFT', 'GOOG', 'TSLA', 'AMZN', '']

SAVE_DIR = 'data'

CHART_DIR = os.path.join(SAVE_DIR, 'charts')

LABEL_DIR = os.path.join(SAVE_DIR, 'labels')


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

    ticker_symbol = None
    if isinstance(df_plot.columns, pd.MultiIndex):
        
        for col_tuple in df_plot.columns:
            if isinstance(col_tuple, tuple) and len(col_tuple) > 1:
                ticker_symbol = col_tuple[1]
                break
        if ticker_symbol is None:
            raise ValueError("Ticker symbol could not be determined from MultiIndex columns in save_chart.")
    else:
        
        raise ValueError("DataFrame columns are not a MultiIndex as expected in save_chart.")


    new_df_for_mpf = pd.DataFrame(index=df_plot.index)
    new_df_for_mpf['Open'] = df_plot[('Open', ticker_symbol)]
    new_df_for_mpf['High'] = df_plot[('High', ticker_symbol)]
    new_df_for_mpf['Low'] = df_plot[('Low', ticker_symbol)]
    new_df_for_mpf['Close'] = df_plot[('Close', ticker_symbol)]
    
    if ('Volume', ticker_symbol) in df_plot.columns:
        new_df_for_mpf['Volume'] = df_plot[('Volume', ticker_symbol)]

    for col in ['Open', 'High', 'Low', 'Close']:
        new_df_for_mpf[col] = new_df_for_mpf[col].astype(float)
    if 'Volume' in new_df_for_mpf.columns:
         new_df_for_mpf['Volume'] = new_df_for_mpf['Volume'].astype(float)

    new_df_for_mpf.index = pd.to_datetime(new_df_for_mpf.index)


    mpf.plot(new_df_for_mpf, type='candle', style='charles',
             savefig=os.path.join(CHART_DIR, f"{fname}.png"))


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
        
        df = yf.download(ticker, period='1y', interval='1d')
        
        df.dropna(inplace=True)

        for i in range(10, len(df)):
            
            window = df.iloc[i-10:i]
            
            label = detect_patterns(df, i)
            if label != "no pattern":
                save_chart(window, count, label)
                count += 1
                
                if count >= len(TICKERS) * num_per_ticker:
                    print(f"Saved {count} charts")
                    return

if __name__ == "__main__":
    generate_dataset()