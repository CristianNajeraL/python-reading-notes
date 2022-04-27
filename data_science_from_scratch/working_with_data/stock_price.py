"""
How to create dataclass objects
"""

import datetime
from dataclasses import dataclass
from typing import NoReturn


@dataclass
class StockPrice:
    """
    Data class to represent stock prices
    """
    symbol: str
    date: datetime.date
    closing_price: float

    def __post_init__(self) -> NoReturn:
        """
        Create properties after instantiate the class
        :return: NoReturn
        :rtype: NoReturn
        """
        self.is_high_tech = self.symbol in ['MSFT', 'GOOG', 'FB', 'AMZN', 'AAPL']
