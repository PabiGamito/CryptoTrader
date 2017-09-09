class Portfolio(object):

    def __init__(self, initial_holdings = {'BTC': 1} ):
        """
        initial_holdings: should be a dictionary with coin symbol as key
        and amount of specific coin held in portfolio as value
        """
        self.holdings = initial_holdings

    def trade(self, from_coin, to_coin, amount, exchange_rate):
        """
        from_coin:     coin that is being sold/traded out of portfolio
        to_coin:       coin that is being bought/traded into portfolio)
        amount:        amount of coin that is being sold/traded out of portfolio
        exchange_rate: from_coin/to_coin, how much is 1 *from_coin* worth in *to_coin*
                       or how much does 1 *from_coin* give you in *to_coin*

        Returns the new holdings
        """

        if self.holdings[from_coin] >= amount:
            self.holdings[from_coin] -= amount
            if to_coin in self.holdings:
                self.holdings[to_coin] += amount * exchange_rate
            else:
                self.holdings[to_coin] = amount * exchange_rate
        else:
            raise RuntimeError('Amount greater than available balance.')

        return self.holdings

    def buy(self, buying_coin, from_coin, amount, exchange_rate):
        return self.trade(from_coin, buying_coin, amount, exchange_rate)

    def sell(selling_coin, to_coin, amount, exchange_rate):
        return self.trade(selling_coin, to_coin, amount, 1/exchange_rate)

    def deposit(self, coin, amount):
        """
        Return the balance remaining after depositing *amount* of *coin*.
        """
        self.holding[coin] += amount
        return self.holding[coin]

    def withdraw(self, coin, amount):
        """
        Return the balance remaining after withdrawing *amount* of *coin*
        """
        if amount > self.holding[coin]:
            raise RuntimeError('Amount greater than available balance.')
        self.holding[coin] -= amount
        return self.balance