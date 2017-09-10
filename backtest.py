class Backtest(object):

    def __init__(self, initial_holdings = {'BTC': 1} ):
        """
        initial_holdings: should be a dictionary with coin symbol as key and
                          amount of specific coin held in portfolio as value
        """
        self.portfolio = Portfolio(initial_holdings)

    def run(self, buy_function, sell_function, train_data, test_data):
        """
        buy_function:  should be a function that returns true when buy condition
                       is met
        sell_function: should be a function that returns true when sell
                       condition is met
        train_data &:  should be a pandas dataframes with date as index and with
        test_data      at least a price column
        """
        self.buy_function = buy_function
        self.sell_function = sell_function
