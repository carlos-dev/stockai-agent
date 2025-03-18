import json
import os
from datetime import datetime
import  yfinance as yf
from crewai import Agent, Task, Crew, Process
from crewai.tools import tool
from langchain.tools import Tool
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchResults
import streamlit as st


@tool("Yahoo Finance Tool")
def fetch_stock_price(ticker: str):
    """
    This tool fetches stock price information for a given ticker symbol from Yahoo Finance
    """
    
    try:
        stock = yf.download(ticker, start='2023-01-01', end=pd.Timestamp.today().strftime('%Y-%m-%d'))
        
        return stock
    except Exception as e:
        return f"Error fetching data for {ticker}: {str(e)}"


os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
llm = ChatOpenAI(modal='gpt-3.5-turbo')

stock_price_analyst = Agent(
  role="Senior stock price analyst",
  goal="Find the {ticker} stock price and analyze trends",
  backstory="""
    You're a highly experienced stock price analyzing the pruce os an specific stock and make predictions about its future price and trends.
  """,
  llm=llm,
  max_iter=5,
  memory=True,
  verbose=True,
  allow_delegation=False,
  tools=[fetch_stock_price]
)

get_stock_price_task = Task(
  description="Analyze the stock {ticker} price history and create a trend analyses of up, down or sideways",
  expected_output="""
    Specify the currente trend stock price - up, down or sideways.
    eg. stock = 'AAPL', price UP
  """,
  agent=stock_price_analyst,
)

## importando a tool de search
@tool("DuckDuckGo Search Tool")
def search_news():
    """
    This tool searches the web for news articles related to a given query using DuckDuckGo.
    """
    search_tool = DuckDuckGoSearchResults(backend='news', num_results=10)
    return search_tool

news_analyst = Agent(
  role="Senior news analyst",
  goal="""
    Create a short summary of the market news related to the stock {ticker} company. Specify the current trend - up, down or sideways with the news context. For each request stock, specify a number between 0 and 100, where 0 is extreme fear and 100 is extreme greed.
  """,
  backstory="""
    You're a highly experienced in analyzing the market trends and news have tracked assest for more than 10 years. 
    You're also master level analytics in the tradicional markets and have deep understading of human psychology.
    You understand news, theirs titles and information, but you look at those with a healthy skepticism.
    You consider also the source of the news articles
  """,
  llm=llm,
  max_iter=10,
  memory=True,
  verbose=True,
  allow_delegation=False,
  tools=[search_news]
)

get_news_task = Task(
  description=f"""take the stock and always include BTC to is (if not request).
    Use the search tool to each one individually.
    The current date is {datetime.now()}.
    Compose the results into a helpful report
  """,
  expected_output="""
    A summary of the overall market and one sentence summary for each request asset.
    Include a feat/greed score for each asset based on the news. Use format:
    <STOCK ASSET>
    <SUMMARY BASED ON NEWS>
    <TREND PREDICTION>
    <FEAR/GREED SCORE>
    """,
  agent=news_analyst,
)

stock_analyst_writer = Agent(
  role="Senior stock analyst writer",
  goal="""
    Analyze the trends price and news and write an insightful compelling and informative 3 paragraph long newsletter based on the stock report and price trend.
  """,
  backstory="""
    You're a widely accepted as the best stock analyst in the market. You understand complex concepts and create compelling stories and narratives that resonate with wider audiences.
    You understand macro factors and combine multiples theories - eg. cycle theory and fundamental analysis. You're able to hold multiple opinions when analyzing anything.
  """,
  llm=llm,
  max_iter=5,
  memory=True,
  verbose=True,
  allow_delegation=True,
  tools=[search_news]
)

write_newsletter_task = Task(
  description="""Use the stock price trend and the stock news report to create an analyses and write the newsletter about the {ticker} company that is brief and highliths the most important points.
  Focus on the stock price trend, news and fear/greed score. What are the near future considerations?
  Include the previous analyses of stock trend and news summary.
  """,
  expected_output="""An eloquent 3 paragraphs newsletter in portuguese formated as markdown in an wasy readble manner. It should contains:

  - 3 bullets executive summary
  - Introduction - set the overhall picture and spike up the interest
  - main part provides the most meat of the analysis inlcudind the news summary and feed/greed scores
  - summary - key facts and concrete future trend prediction - up, down or sideways
  """,
  context=[get_stock_price_task, get_news_task],
)

crew = Crew(
  agents=[stock_price_analyst, news_analyst, stock_analyst_writer],
  tasks=[get_stock_price_task, get_news_task, write_newsletter_task],
  verbose=True,
  process=Process.hierarchical,
  full_output=True,
  share_crew=False,
  manager_llm=llm,
  max_iter=15,
)

with st.sidebar:
  st.header("Enter the ticker symbol")

  with st.form(key='research_form'):
    topic = st.text_input("Select the ticker")
    submit_button = st.form_submit_button(label='Run research')

if submit_button:
  if not topic:
     st.error("Please enter a ticker symbol")
  else:
     results = crew.kickoff(inputs={"ticker": topic})
     st.subheader("Results")
     st.write(results.raw)



