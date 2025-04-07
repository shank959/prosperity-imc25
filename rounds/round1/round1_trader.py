from ...utils.datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Tuple

class Trader:
    # we are looking to run two different strategies for rainforest resin
    # 1. market make - where we have to figure out optimal spread around a an optimal price and an optimal volume
    # 2. market take - where we have to figure out the what prices are good for us

    def kelp_fair(self, state: TradingState) -> float:
        pass

    def market_make(
            self,
            state: TradingState,
            product: str,
            spread: float,
            volume: int,
            ) -> List[Order]:
        # have a price
        # add spread
        # post orders on either side
        pass

    def market_take(
            self,
            state: TradingState,
            product: str,
            spread: float,
            volume: int,
            ) -> List[Order]:
        # have a max buy and min sell price
        # check if order depth has any orders
        pass


    def run(self, state: TradingState) -> Tuple[List[Order], int, str]:
        result = []
        conversions = 0
        traderData = ""

        # we want to both market make and market take

        # some logic for rainforest resin

        # some logic for kelp
        kelp_fair = 0

        # some logic for squid ink
        squid_ink_fair = 0

        return result, conversions, traderData
