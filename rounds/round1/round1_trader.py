from ...utils.datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import string
import jsonpickle
import numpy as np
import math

class Trader:
    # we are looking to run two different strategies for rainforest resin
    # 1. market make - where we have to figure out optimal spread around a an optimal price and an optimal volume
    # 2. market take - where we have to figure out the what prices are good for us

    # for kelp
    def market_make(self, state: TradingState) -> List[Order]:
        pass

    def market_take(self, state: TradingState) -> List[Order]:
        pass
    def run(self, state: TradingState) -> Tuple[List[Order], int, dict]:
        # we want to both market make and market take