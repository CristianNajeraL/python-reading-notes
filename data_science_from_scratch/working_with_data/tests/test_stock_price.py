"""
Testing stock price data object
"""

import datetime
from unittest import TestCase

from ..stock_price import StockPrice as sp


class TestStockPrice(TestCase):
    """
    This class contains tests for stock price data object
    """

    def test_stock_price_data_object(self):
        """Successfully test"""
        price = sp(symbol='MSFT', date=datetime.date(2018, 12, 14), closing_price=106.03)
        self.assertEqual(price.symbol, 'MSFT')
        self.assertEqual(price.closing_price, 106.03)
        self.assertEqual(price.is_high_tech, True)
