import pandas as pd
import numpy as np

def get_data(file_name):
  data = pd.read_csv(file_name, sep='\t')
  # Keep just needed data for new dataframe
  # data = data[['close', 'date']].set_index('date')
  return data

def get_file_name(pair, period = 86400):
  return 'data/' + pair + '-p' + str(period) + '.csv'

def pair_data_frame(pair, period = 86400):
  return get_data(get_file_name(pair, period))

def generate_data_frame(pair, columns = ['date', 'close'], period = 86400):
  df = get_data(get_file_name(pair, period))
  # Keep just needed data for new dataframe
  df = df[ columns ]

  result = df.sort_values('date')#.set_index('date')
  # Convert all result data to floats
  result = result.astype(float)
  return result

def generate_data_frame_of_multiple_pairs(pairs, column = 'close', period = 86400):
  """
  Function take a list of pairs to generate a pandas data frame for,
  and the name of the column to keep and returns a pandas data frame
  with all data for the selected column for all selected pairs
  ex: generate_data_frame_of_multiple_pairs(["BTC_ETH, BTC_USD"], 'close') will return:
  date      | BTC_ETH             | BTC_USD
  timestamp | btc_eth close price | btc_usd close price
  """

  # TODO: Check len(pairs) > 1 else warn to use generate_data_frame function for single pair

  # Initiate DataFrame
  df = pd.DataFrame()
  # Initiate dictionary to store dataframes

  for pair in pairs:
    df_temp = get_data(get_file_name(pair, period))
    # Keep just needed data for new dataframe
    columns = ['date', column]
    df_temp = df_temp[ columns ]
    # Rename close column to pair
    df_temp = df_temp.rename(columns={'close': pair})
    if df.empty:
      # initialize dataframe (df) if df is empty with first data
      df = df_temp
    else:
      df = df.merge(df_temp, on='date', how='outer')

  result = df.sort_values('date')#.set_index('date')
  # Convert all result data to floats
  result = result.astype(float)
  return result

def break_even_amount(buy_price, amount, target_change, trade_fee = 0):
  """
  This function take a buy order price, the amount to buy
  and a target price to sell as a relative change from buy price
  and returns the amount to sell at the target price to break even
  """
  buy_value = buy_price * amount
  buy_cost = buy_value * ( 1 + trade_fee )
  traget_sell_price = buy_price * ( 1 + target_change )
  sell_amount = (buy_cost / traget_sell_price) / ( 1 - trade_fee)

  return sell_amount

def value_after_trade(price, amount, fee):
  return amount * price * (1 - fee)

def local_turning_points(df, n=3):
  minimas = []
  maximas = []
  for i in range(1, df.shape[0] - 1):
    lp = float(df.iloc[i - 1])
    p = float(df.iloc[i])
    np = float(df.iloc[i + 1])

    if lp > p and p < np:
      minimas.append(p)
    elif lp < p and p > np:
      maximas.append(p)

  return maximas, minimas

def support_resistance(df, range = 0.01):
  maximas, minimas = local_turning_points(df)
  turning_points = maximas + minimas
  sr_points = {}

  for tp in turning_points:
    close_points = []
    for i, tp_compare in enumerate(turning_points):
      if tp*(1-range) < tp_compare and tp_compare < tp*(1+range):
        close_points.append(tp_compare)
        del turning_points[i]

    avg_sr_price = sum(close_points)/float(len(close_points))
    sr_points[avg_sr_price] = len(close_points)


  # sort sr_points by value (amount of time it touches support point)
  sr_points = sorted(sr_points.items(), key=lambda x: x[1], reverse=True)

  return sr_points


def plot_support_resistance(spt_res):
  for sr_point in spt_res:
    if sr_point[1] > 1:
      plt.axhline(y=sr_point[0],xmin=0,xmax=3,c="blue",linewidth=0.5,zorder=0)


def william_vix_fix(df, look_back_period=22):
  """
  df: should be a pandas dataframe ordered in ascending date with close price (close), low price (low)
  look_back_period: should be an integer of how many datapoints to look to get the latest high to determine vix fix value
  returns a data frame with date and vix_fix value for each corresponding date
  """

  # Initialize dataframe
  william_vix_fix_df = pd.DataFrame(columns=('date', 'vix_fix'))

  i = 0
  # Highest closing price in most recent 'look_back_period' trading day
  look_back_period_close = []
  for row in df.itertuples():
    i += 1
    close = row.close
    look_back_period_close.append(close)

    if i >= look_back_period:
      highest_close = max(look_back_period_close)
      look_back_period_close.pop(0)
      date = row.date
      low = row.low
      william_vix_fix = ( ( (highest_close - low) / highest_close) * 100 ) # + 50
      william_vix_fix_df.loc[i-look_back_period] = [date, william_vix_fix]

  return william_vix_fix_df

def plot_buy_signal(ax, x, y):
  ax.plot(x, y, marker="^", linestyle='-', color='g')

def plot_sell_signal(ax):
  ax.plot(x, y, marker='v', linestyle='-', color='r')

def pd_ema(df, period):
  alpha = 2.0/(period+1)
  ma = df.rolling(period).mean()
  return ma.ewm(alpha=alpha, adjust=False).mean()

def get_poloniex_data_from_api(pair, period, start, end=9999999999):
  from poloniex import Poloniex
  polo = Poloniex()
  data = polo.returnChartData(pair, period, start, end)
  return pd.DataFrame.from_dict(data).astype(float)

def update_poloniex_data_from_api(pair, period):
  # Find file
  # If it exists get latest update ts and add new data
  # Else get all data and create file
  pass

"""
Testing functions
"""

# from matplotlib import pyplot as plt
# from scipy.signal import savgol_filter as smooth

# f, ax = plt.subplots(2, sharex=True)

# import time
# ts_now = time.time()
# ts = ts_now - 90*24*60*60

# # periods: 5m: 300, 15m: 900, 30m: 1800, 2h: 7200, 4h: 14400, and 1d: 86400
# df = get_poloniex_data_from_api('BTC_ETH', 7200, ts)

# price_data = df[['date', 'close']].set_index('date')
# price_data.plot(ax = ax[0])
# # plot ema
# pd_ema = pd_ema(price_data, 200)
# pd_ema.columns = ['ema']
# pd_ema.plot(ax = ax[0])


# william_vix_fix = william_vix_fix(df[['date', 'close', 'low']]).set_index('date')
# william_vix_fix.plot(ax = ax[1])

# plt.show()


# usdt_btc_close = generate_data_frame("USDT_BTC", ['date', 'close'], 300).set_index('date').tail(1000)
# usdt_btc_close.plot(ax = ax[0])
# pd_ema(usdt_btc_close, 200).plot(ax = ax[0])

# df = generate_data_frame("USDT_BTC", ['date', 'close', 'low'], 300).tail(10000)
# william_vix_fix = william_vix_fix(df, 50).set_index('date')
# william_vix_fix.plot(ax = ax[1])



# plt.show()



# sr = support_resistance(df)

# print sr

# df.plot()
# plot_support_resistance(sr)

# get_local_turning_points(df)

# plt.show()


