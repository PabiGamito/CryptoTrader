from Cryptotrader.functions import *
from Cryptotrader.portfolio import *

initial_holdings = {'USDT': 10000}
portfolio = Portfolio(initial_holdings)

print portfolio.buy("BTC", "USDT", 1, 3800)

def buy(date, vix_fix_df):
  """
  Function returns true if buy condition is met
  vix_fix_df: pandas data frame with date in ts as index
  """
  # Stop reversal of vix_fix above 60
  vix_fix_df[date]

def sell(df, date):
  """
  Function returns true if sell condition is met
  """