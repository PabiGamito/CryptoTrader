import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Cryptotrader.functions import *

""" Variables """
import time
ts_now = int(time.time())
ts = ts_now - 3*365*24*60*60
period = 7200
df = get_poloniex_data_from_api("BTC_ETH", period, ts)

df.to_pickle('df.pkl')
df = pd.read_pickle('df.pkl')
"""
Generate necessary dataframes
"""

# take only 10000 last datapoints
# df = df.tail(100000)

shift_amount = 12
df['target_close'] = df['close'].shift(-shift_amount)

# df['last_close'] = df['close'].shift(1)
# df['6h_close'] = df['close'].shift(12)
# df['12h_close'] = df['close'].shift(24)
# df['24h_close'] = df['close'].shift(48)

# df['24h_change'] = (df['close'] - df['24h_close']) / 100
# df['24h_change'] = (df['24h_change']).shift(1) #24h change of last period

# vix_fix_look_back_period = 22
# william_vix_fix = william_vix_fix(df, vix_fix_look_back_period).set_index('date')

# print(df.set_index('date').head())
# print(william_vix_fix.head())

# df = df[vix_fix_look_back_period:].set_index('date')

# frames = [df, william_vix_fix]
# df['vix'] = william_vix_fix['vix_fix']
# df['last_vix'] = df['vix'].shift(1)

df = df[:-shift_amount].set_index('date')

# df.to_pickle('data.pkl')

# df = pd.read_pickle('data.pkl')

# print( "Train data y", y_train.head() )
# print( "Test data y", y_test.head() )

value_to_predict = 'target_close'

def generate_numpy_df(df):
    data_list = []
    for date, row in df.T.iteritems():
        data = np.asarray([
            df.loc[date, 'volume'],
            df.loc[date, 'low'],
            df.loc[date, 'high'],
            df.loc[date, 'close']
        ])
        data_list.append(data)
    numpy_df = np.asarray(data_list)
    return numpy_df

"""
Random Forest Regressor
"""
def random_forest_regressor(numpy_df_train, numpy_df_test, y_train, y_test):
    from treeinterpreter import treeinterpreter as ti
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import classification_report,confusion_matrix

    rf = RandomForestRegressor(verbose=True)
    rf.fit(numpy_df_train, y_train)

    print('Freature Importance ', rf.feature_importances_)

    print('Generating predictions')
    prediction, bias, contributions = ti.predict(rf, numpy_df_test)
    print('Predictions generated')

    idx = test.index
    predictions_df = pd.DataFrame(data=prediction[0:], index = idx, columns=['predicted_value'])
    return predictions_df

def plotdata(predictions_df, y_test):
    # Plot data
    ax = predictions_df.plot(title='Random Forest predicted prices')
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    fig = y_test.rename(columns={value_to_predict: "actual_price"}).plot(ax = ax).get_figure()
    # fig.savefig("graphs/random forest without smoothing.png")

    plt.show()

train_start_date = df.index[0]
train_end_date = df.index[ int(len(df)*0.90) ]
test_start_date = df.index[ int(len(df)*0.90) + 1 ]
# test_end_date = -1

train = df.ix[train_start_date : train_end_date]
test = df.ix[test_start_date : ]

y_train = pd.DataFrame(train[value_to_predict]) # or use WeightedAverage
y_test = pd.DataFrame(test[value_to_predict])

# Get training dataframe and testing dataframe
numpy_df_train = generate_numpy_df(train)
numpy_df_test = generate_numpy_df(test)
predictions_df = random_forest_regressor(numpy_df_train, numpy_df_test, y_train, y_test)

plotdata(predictions_df, y_test)

loop_df = df[predictions_df.index[0]: ].copy()
loop_df['predicted_value'] = predictions_df['predicted_value'].astype(float)

trade_oportunities = 0
successful_trades = 0
failed_trades = 0

for date, row in loop_df.iterrows():
    predicted_future_price = row['predicted_value']
    future_price = row['target_close']
    close = row['close']
    predicted_rel_change = (predicted_future_price - close) / 100
    rel_change = (future_price - close) / 100

    if predicted_rel_change > 0.5/100:
        trade_oportunities += 1
        print("Predicted rel_change", predicted_rel_change)
        print("Actual rel_change", rel_change)
        print("===========")
    elif predicted_rel_change < -0.5/100:
        trade_oportunities += 1

print(trade_oportunities)

"""
MLP Classifier
"""
'''
def offset_value(test_start_date, test, predictions_df):
    from datetime import datetime, timedelta
    temp_date = test_start_date
    average_last_5_days_test = 0
    average_upcoming_5_days_predicted = 0
    total_days = 10
    period = test.index[1] - test.index[0]
    for i in range(total_days):
        average_last_5_days_test += test.loc[temp_date, 'close']
        temp_date = temp_date + period
    average_last_5_days_test = average_last_5_days_test / total_days

    temp_date = test_start_date
    for i in range(total_days):
        average_upcoming_5_days_predicted += predictions_df.loc[temp_date, 'prices']
        temp_date = temp_date + period
    average_upcoming_5_days_predicted = average_upcoming_5_days_predicted / total_days
    difference_test_predicted_prices = average_last_5_days_test - average_upcoming_5_days_predicted
    return difference_test_predicted_prices

from sklearn.neural_network import MLPClassifier
from datetime import datetime, timedelta

prediction_list = []
for i in range(10):
    # Splitting the training and testing data in 10 datasets
    train_start_date = df.index[i * int(len(df)/10)]
    train_end_date = df.index[ i * int(len(df)/10) + int(len(df)/10 * 0.8) ]
    test_start_date = df.index[ i * int(len(df)/10) + int(len(df)/10 * 0.8) + 1]
    test_end_date = df.index[(i + 1) * int(len(df)/10)]
    train = df.ix[train_start_date : train_end_date]
    test = df.ix[test_start_date:test_end_date]

    # Calculating the sentiment score
    numpy_df_train = generate_numpy_df(train)
    numpy_df_test = generate_numpy_df(test)

    # Generating models
    mlpc = MLPClassifier(hidden_layer_sizes=(100, 200, 100), activation='relu', solver='lbfgs', alpha=0.005, learning_rate_init = 0.001, shuffle=False)
    # NOTE: This can only be integer or string
    y = np.asarray(train['close'].astype(int)) # or use WeightedAverage
    mlpc.fit(numpy_df_train, y)
    prediction = mlpc.predict(numpy_df_test)

    prediction_list.append(prediction)
    #print train_start_date + ' ' + train_end_date + ' ' + test_start_date + ' ' + test_end_date
    idx = test.index
    #print year
    predictions_df_list = pd.DataFrame(data=prediction[0:], index = idx, columns=['prices'])

    difference_test_predicted_prices = offset_value(test_start_date, test, predictions_df_list)
    # Adding offset to all the advpredictions_df price values
    predictions_df_list['prices'] = predictions_df_list['prices'] + difference_test_predicted_prices
    predictions_df_list

    # Smoothing the plot
    # predictions_df_list['ewma'] = pd.ewma(predictions_df_list["prices"], span=20)
    # predictions_df_list['actual_value'] = test['close']
    # predictions_df_list['actual_value_ewma'] = pd.ewma(predictions_df_list["actual_value"], span=20)
    # # Changing column names
    # predictions_df_list.columns = ['predicted_price', 'average_predicted_price', 'actual_price', 'average_actual_price']
    # predictions_df_list.plot()
    # predictions_df_list_average = predictions_df_list[['average_predicted_price', 'average_actual_price']]
    # predictions_df_list_average.plot()

    # predictions_df_list.show()


mlpc = MLPClassifier(hidden_layer_sizes=(100, 200, 50), activation='relu', solver='lbfgs', alpha=0.005, learning_rate_init = 0.001, shuffle=False)

# checking the performance of training data itself
prediction, bias, contributions = ti.predict(rf, numpy_df_test)
idx = test.index
predictions_df = pd.DataFrame(data=prediction[0:], index = idx, columns=['predicted_price'])
# Plot data
ax = predictions_df.plot(title='MLP Classifier predicted prices')
ax.set_xlabel("Dates")
ax.set_ylabel("Stock Prices")
fig = test['close'].rename(columns={"close": "actual_prices"}).plot(ax = ax).get_figure()

plt.show()
'''
