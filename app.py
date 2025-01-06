import openai
from phi.agent import Agent
from phi.model.openai import OpenAIChat
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
from dotenv import load_dotenv
from phi.tools.csv_tools import CsvTools
from phi.tools.openbb_tools import OpenBBTools
from phi.tools.newspaper4k import Newspaper4k
from phi.model.groq import Groq
import os

# Load environment variables from .env file
load_dotenv()

# Set API keys
openai.api_key = os.getenv("OPENAI_API_KEY")
Groq.api_key = os.getenv("GROQ_API_KEY")

# Define the agents with appropriate roles and tools
# 1. Agent to fetch real-time stock data using OpenBB
real_time_agent = Agent(
    name="Real-Time Stock Data Agent",
    model=Groq(id="llama3-groq-70b-8192-tool-use-preview"),
    role="Fetch real-time stock data from the portfolio",
    tools=[OpenBBTools()],
    instructions=[
        "Fetch the real-time stock data for the portfolio items.",
        "Provide data in an organized tabular format."
    ],
    show_tools_calls=True,
    markdown=True,
)

# 2. Agent to search the web for additional stock-related information
web_search_agent = Agent(
    name="Web Search Agent",
    model=Groq(id="llama3-groq-70b-8192-tool-use-preview"),
    role="Search the web for information related to stocks in the portfolio",
    tools=[DuckDuckGo()],
    instructions=[
        "Search and gather relevant information about the stocks in the portfolio.",
        "Always include credible sources for the information provided."
    ],
    show_tools_calls=True,
    markdown=True,
)

# 3. Finance agent to retrieve stock fundamentals, price, and analyst recommendations
finance_agent = Agent(
    name="Finance Data Agent",
    role="Retrieve stock fundamentals, prices, and analyst recommendations",
    model=Groq(id="llama3-groq-70b-8192-tool-use-preview"),
    tools=[
        YFinanceTools(
            stock_price=True,
            analyst_recommendations=True,
            stock_fundamentals=True,
            company_news=False
        ),
        CsvTools(read_csvs=["portfolio.csv"])
    ],
    instructions=[
        "Fetch stock fundamentals, current prices, and analyst recommendations for the portfolio.",
        "Use tables to present the data in a clear and concise format."
    ],
    show_tools_calls=True,
    markdown=True,
)

# 4. Agent to summarize news related to the stocks
news_agent = Agent(
    name="News Summarization Agent",
    role="Summarize news articles related to the stocks in the portfolio",
    model=Groq(id="llama3-groq-70b-8192-tool-use-preview"),
    tools=[Newspaper4k()],
    instructions=[
        "Fetch and summarize recent news articles related to the stocks in the portfolio.",
        "Ensure the summaries are concise and provide links to the original articles."
    ],
    show_tools_calls=True,
    markdown=True,
)

# Define the multi-agent system
portfolio_team = Agent(
    team=[real_time_agent, web_search_agent, finance_agent, news_agent],
    instructions=[
        "Coordinate between agents to generate a comprehensive portfolio report.",
        "Ensure the report includes real-time data, web insights, stock fundamentals, and news summaries."
    ],
    show_tools_calls=True,
    markdown=True,
)

# Generate the portfolio report
portfolio_team.print_response(
    ''' 
    date	tic	open	high	low	close	volume
01-02-2013	DOW	13104.2998	13412.70996	13104.2998	13412.5498	161430000
01-03-2013	DOW	13413.00977	13430.59961	13358.2998	13391.36035	129630000
01-04-2013	DOW	13391.0498	13447.11035	13376.23047	13435.20996	107590000
01-07-2013	DOW	13436.12988	13436.12988	13343.32031	13384.29004	113120000
01-08-2013	DOW	13377.41992	13377.41992	13293.12988	13328.84961	129570000

SUMMERIZE THE ABOVE PORTFOLIO DATA USING ALL THE AGENTS
''',
    stream=True
)


