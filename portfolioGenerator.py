import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def get_stock_data():
    """Generate portfolio data for Indian stocks"""
    # Define stock symbols
    symbols = [
        'GOOG',  # Tata Motors
        'TESLA',       # IREDA
        'AMZN'        # IRCTC
    ]
    
    # Get data for the last 30 days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    data = []
    for symbol in symbols:
        try:
            stock = yf.Ticker(symbol)
            hist = stock.history(start=start_date, end=end_date)
            
            for date, row in hist.iterrows():
                data.append({
                    'date': date.strftime('%Y-%m-%d'),  # Correct date format
                    'tic': symbol,
                    'open': row['Open'],
                    'high': row['High'],
                    'low': row['Low'],
                    'close': row['Close'],
                    'volume': int(row['Volume'])
                })
        except Exception as e:
            print(f"Error fetching data for {symbol}: {str(e)}")
    
    df = pd.DataFrame(data)
    df = df.sort_values(['date', 'tic'])
    return df

if __name__ == "__main__":
    df = get_stock_data()
    df.to_csv('portfolio.csv', sep='\t', index=False)
    print("\nSample of generated portfolio data:")
    print(df.head(15))