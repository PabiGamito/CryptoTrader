import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Cryptotrader.functions import *

""" Variables """
# import time
# ts_now = int(time.time())
# ts = ts_now - 365*24*60*60
# period = 300
# df = get_poloniex_data_from_api("USDT_BTC", period, ts)
# df.to_pickle("USDT_BTC.pkl")  # where to save it, usually as a .pkl

percentage_of_data_to_train = 0.8

"""
Generate necessary dataframes
"""
df = pd.read_pickle("USDT_BTC.pkl")

# take only 10000 last datapoints
df = df.tail(10000)

vix_fix_look_back_period = 22
william_vix_fix = william_vix_fix(df, vix_fix_look_back_period).set_index('date')

# print(df.set_index('date').head())
# print(william_vix_fix.head())

df = df[vix_fix_look_back_period-1:].set_index('date')

frames = [df, william_vix_fix]
df['vix'] = william_vix_fix['vix_fix']

# print( "Train data y", y_train.head() )
# print( "Test data y", y_test.head() )

def generate_numpy_df(df):
    data_list = []
    for date, row in df.T.iteritems():
        data = np.asarray([df.loc[date, 'volume'], df.loc[date, 'vix']])
        data_list.append(data)
    numpy_df = np.asarray(data_list)
    return numpy_df

"""
Random Forest Regressor
"""
from treeinterpreter import treeinterpreter as ti
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import classification_report,confusion_matrix

train_start_date = df.index[0]
train_end_date = df.index[ int(len(df)*0.8) ]
test_start_date = df.index[ int(len(df)*0.8) + 1 ]
# test_end_date = -1

train = df.ix[train_start_date : train_end_date]
test = df.ix[test_start_date : ]

y_train = pd.DataFrame(train['close']) # or use WeightedAverage
y_test = pd.DataFrame(test['close'])

# Get training dataframe and testing dataframe
numpy_df_train = generate_numpy_df(train)
numpy_df_test = generate_numpy_df(test)

rf = RandomForestRegressor()
rf.fit(numpy_df_train, y_train)

print(rf.feature_importances_)

prediction, bias, contributions = ti.predict(rf, numpy_df_test)

idx = test.index
predictions_df = pd.DataFrame(data=prediction[0:], index = idx, columns=['predicted_price'])

# Plot data
ax = predictions_df.plot(title='Random Forest predicted prices')
ax.set_xlabel("Dates")
ax.set_ylabel("Stock Prices")
fig = y_test.rename(columns={"close": "actual_prices"}).plot(ax = ax).get_figure()
# fig.savefig("graphs/random forest without smoothing.png")

plt.show()

"""
MLP Classifier
"""
from sklearn.neural_network import MLPClassifier
from datetime import datetime, timedelta

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
    mlpc = MLPClassifier(hidden_layer_sizes=(100, 200, 100), activation='relu',
                         solver='lbfgs', alpha=0.005, learning_rate_init = 0.001, shuffle=False) # span = 20 # best 1
    mlpc.fit(numpy_df_train, train['close'])
    prediction = mlpc.predict(numpy_df_test)

    prediction_list.append(prediction)
    #print train_start_date + ' ' + train_end_date + ' ' + test_start_date + ' ' + test_end_date
    idx = pd.date_range(test_start_date, test_end_date)
    #print year
    predictions_df_list = pd.DataFrame(data=prediction[0:], index = idx, columns=['prices'])

    difference_test_predicted_prices = offset_value(test_start_date, test, predictions_df_list)
    # Adding offset to all the advpredictions_df price values
    predictions_df_list['prices'] = predictions_df_list['prices'] + difference_test_predicted_prices
    predictions_df_list

    # Smoothing the plot
    predictions_df_list['ewma'] = pd.ewma(predictions_df_list["prices"], span=20, freq="D")
    predictions_df_list['actual_value'] = test['prices']
    predictions_df_list['actual_value_ewma'] = pd.ewma(predictions_df_list["actual_value"], span=20, freq="D")
    # Changing column names
    predictions_df_list.columns = ['predicted_price', 'average_predicted_price', 'actual_price', 'average_actual_price']
    predictions_df_list.plot()
    predictions_df_list_average = predictions_df_list[['average_predicted_price', 'average_actual_price']]
    predictions_df_list_average.plot()

#     predictions_df_list.show()


mlpc = MLPClassifier(hidden_layer_sizes=(100, 200, 50), activation='relu', solver='lbfgs', alpha=0.005, learning_rate_init = 0.001, shuffle=False)


# checking the performance of training data itself
prediction, bias, contributions = ti.predict(rf, numpy_df_test)
idx = test.index
predictions_df = pd.DataFrame(data=prediction[0:], index = idx, columns=['predicted_price'])
# Plot data
ax = predictions_df.plot(title='Random Forest predicted prices')
ax.set_xlabel("Dates")
ax.set_ylabel("Stock Prices")
fig = train['prices'].rename(columns={"close": "actual_prices"}).plot(ax = ax).get_figure()

plt.show()
