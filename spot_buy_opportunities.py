import re

def poloniex_buy_opportunities():
    buy_pairs = []
    from poloniex import Poloniex
    polo = Poloniex()
    ticker = polo.returnTicker()
    for pair, data in ticker.items():
        percentChange = float(data['percentChange'])
        volume = float(data['baseVolume'])
        if percentChange < -0.1:
            if volume > 1000:
                buy_pairs.append(pair)
            elif percentChange < -0.2:
                buy_pairs.append(pair)

    # Select only pairs trading against BTC
    regex = re.compile('BTC')
    buy_pairs = list( filter(regex.search, buy_pairs) )
    return buy_pairs

def bittrex_buy_opportunities():
    buy_pairs = []
    from Cryptotrader.bittrex import Bittrex
    bittrex = Bittrex(None, None)
    ticker = bittrex.get_market_summaries()['result']
    for data in ticker:
        pair = data['MarketName']
        percentChange = data['Last'] / data['PrevDay'] - 1
        volume = data['BaseVolume']
        if percentChange < -0.1:
            if volume > 1000:
                buy_pairs.append(pair)
            elif percentChange < -0.2:
                buy_pairs.append(pair)

    # Select only pairs trading against BTC
    regex = re.compile('BTC')
    buy_pairs = list( filter(regex.search, buy_pairs) )
    return buy_pairs

print(poloniex_buy_opportunities())

print(bittrex_buy_opportunities())
