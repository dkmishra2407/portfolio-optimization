import os
import io
import pandas as pd
from typing import List, Dict
from datetime import datetime
import yfinance as yf
from dotenv import load_dotenv
from openai import OpenAI
import numpy as np

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class PortfolioData:
    def __init__(self, date: str, ticker: str, open_price: float, high: float, low: float, close: float, volume: int):
        self.date = datetime.strptime(date, '%Y-%m-%d')  # Parse date in correct format
        self.ticker = ticker
        self.open = open_price
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume

class MarketAnalyzer:
    def get_stock_data(self, ticker: str, start_date: datetime, end_date: datetime) -> Dict:
        """Fetch stock data using yfinance"""
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d')
            )
            return {
                "data": hist,
                "info": stock.info
            }
        except Exception as e:
            print(f"Error fetching data for {ticker}: {str(e)}")
            return None

    def analyze_trends(self, data: pd.DataFrame) -> Dict:
        """Analyze price trends and calculate technical indicators"""
        try:
            returns = data['Close'].pct_change().mean()
            volatility = data['Close'].pct_change().std()
            volume_trend = data['Volume'].mean()
            price_trend = "Upward" if data['Close'].iloc[-1] > data['Close'].iloc[0] else "Downward"
            
            analysis = {
                "returns": returns,
                "volatility": volatility,
                "volume_trend": volume_trend,
                "price_trend": price_trend
            }
            return analysis
        except Exception as e:
            print(f"Error analyzing trends: {str(e)}")
            return {
                "returns": 0.0,
                "volatility": 0.0,
                "volume_trend": 0,
                "price_trend": "Unknown"
            }

class NewsAnalyzer:
    def get_news(self, ticker: str) -> List[Dict]:
        """Get news articles for a given ticker"""
        try:
            stock = yf.Ticker(ticker)
            news = stock.news
            return [{"title": n.get("title", ""), "summary": n.get("summary", "")} for n in news[:5]]
        except Exception as e:
            print(f"Error fetching news for {ticker}: {str(e)}")
            return []

    def analyze_sentiment(self, text: str) -> Dict:
        """Analyze sentiment using OpenAI's updated API"""
        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "Analyze the sentiment of this text and respond with only a number between -1 (very negative) and 1 (very positive)."},
                    {"role": "user", "content": text}
                ]
            )
            sentiment_score = float(response.choices[0].message.content.strip())
            return {"sentiment": sentiment_score}
        except Exception as e:
            print(f"Error analyzing sentiment: {str(e)}")
            return {"sentiment": 0.0}

class PortfolioAnalyzer:
    def __init__(self):
        self.market_analyzer = MarketAnalyzer()
        self.news_analyzer = NewsAnalyzer()

    def load_portfolio_data(self, data: str) -> List[PortfolioData]:
        """Parse and load portfolio data from string input"""
        try:
            df = pd.read_csv(io.StringIO(data), sep='\t')
            return [
                PortfolioData(
                    date=row['date'],
                    ticker=row['tic'],
                    open_price=float(row['open']),
                    high=float(row['high']),
                    low=float(row['low']),
                    close=float(row['close']),
                    volume=int(row['volume'])
                )
                for _, row in df.iterrows()
            ]
        except Exception as e:
            print(f"Error loading portfolio data: {str(e)}")
            return []

    def analyze_portfolio(self, portfolio_data: List[PortfolioData]) -> Dict:
        results = {
            "market_analysis": {},
            "news_analysis": {}
        }

        for entry in portfolio_data:
            market_data = self.market_analyzer.get_stock_data(
                entry.ticker,
                entry.date,
                datetime.now()
            )
            
            if market_data:
                results["market_analysis"][entry.ticker] = (
                    self.market_analyzer.analyze_trends(market_data["data"])
                )
            
            news = self.news_analyzer.get_news(entry.ticker)
            if news:
                sentiments = [
                    self.news_analyzer.analyze_sentiment(article["title"])["sentiment"]
                    for article in news if article["title"]
                ]
                results["news_analysis"][entry.ticker] = {
                    "articles": news,
                    "sentiment": np.mean(sentiments) if sentiments else 0.0
                }

        return results

    def generate_report(self, analysis_results: Dict) -> str:
        report = f"""
# Portfolio Analysis Report
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Market Analysis
"""
        for ticker, analysis in analysis_results["market_analysis"].items():
            report += f"""
### {ticker}
- Returns: {analysis['returns']:.2%}
- Volatility: {analysis['volatility']:.2%}
- Volume Trend: {analysis['volume_trend']:,.0f}
- Price Trend: {analysis['price_trend']}
"""

        report += "\n## News Analysis\n"
        for ticker, news in analysis_results["news_analysis"].items():
            report += f"""
### {ticker}
- Overall Sentiment: {news['sentiment']:.2f}
- Recent Headlines:
"""
            for article in news["articles"]:
                if article["title"]:
                    report += f"  - {article['title']}\n"

        return report

def main():
    try:
        with open('portfolio.csv', 'r') as f:
            portfolio_data = f.read()
    except FileNotFoundError:
        print("Please run the data generator script first to create portfolio.csv")
        return

    analyzer = PortfolioAnalyzer()
    portfolio_entries = analyzer.load_portfolio_data(portfolio_data)
    
    if not portfolio_entries:
        print("No portfolio data could be loaded.")
        return
        
    analysis_results = analyzer.analyze_portfolio(portfolio_entries)
    report = analyzer.generate_report(analysis_results)
    print(report)

if __name__ == "__main__":
    main()