from datamodel import OrderDepth, UserId, TradingState, Order, ConversionObservation
from typing import List
import string
import jsonpickle
import numpy as np
import math

PARAMETERS = {
    "RAINFOREST_RESIN": {
        "fair_value": 10000,
        "position_limit": 50
    },
    "KELP": {
        "market_makers_size": 15,
        "position_limit": 50,
    },
    "SQUID_INK": {
        "market_makers_size": 15,
        "position_limit": 50,
    }
}


class Trader:

    def __init__(self):
        self.RFR = {
            "position_limit": 50,
            "fair_value": 10000,
        }
        self.KELP = {
            "position_limit": 50,
            "market_makers_size": 15,
            "mPrice": []
        }
        self.SQUID = {
            "position_limit": 50,
            "market_makers_size": 15,
            "mPrice": []
        }
        self.PIC1 = {
            "position_limit": 60,
            "mPrice": []
        }
        self.PIC2 = {
            "position_limit": 100,
            "mPrice": []
        }
        self.CROSSOINT = {
            "position_limit": 250,
            "mPrice": []
        }
        self.JAM = {
            "position_limit": 350,
            "mPrice": []
        }
        self.DJEMBE = {
            "position_limit": 60,
            "mPrice": []
        }

    # INVENTORY MANAGEMENT
    def clearPos(self,
                 orders: List[Order],
                 orderDep: OrderDepth,
                 position: int,
                 positionLimit: int,
                 product: str,
                 buyVolume: int,
                 sellVolume: int,
                 fairVal: float) -> List[Order]:
        positionAfter = position + buyVolume - sellVolume
        fairBid = math.floor(fairVal)
        fairAsk = math.ceil(fairVal)
        #! TODO: TRY OUT DIFFERENT METHODS OF OFLLOADING position
        # fairAsk = fairBid = fair

        buyQuant = positionLimit - (position + buyVolume)
        sellQuant = positionLimit + (position - sellVolume)

        if positionAfter > 0:
            if fairAsk in orderDep.buy_orders.keys():
                clearQuant = min(
                    orderDep.buy_orders[fairAsk], positionAfter)
                #! TODO: SEE IF WE WANT TO OFFLOAD ENTIRE POSITIONS
                sentQuant = min(sellQuant, clearQuant)
                orders.append(Order(product, round(fairAsk), -abs(sentQuant)))
                sellVolume += abs(sentQuant)

        if positionAfter < 0:
            if fairBid in orderDep.sell_orders.keys():
                clearQuant = min(
                    abs(orderDep.sell_orders[fairBid]), abs(positionAfter))
                # clearQuant = abs(positionAfter)
                sentQuant = min(buyQuant, clearQuant)
                orders.append(Order(product, round(fairBid), abs(sentQuant)))
                buyVolume += abs(sentQuant)

        return buyVolume, sellVolume

    # RAINFROEST_RESIN
    def rfr_order(self, order_depth: OrderDepth, position: int) -> List[Order]:
        fv = self.RFR["fair_value"]
        pos_lim = self.RFR["position_limit"]

        orders = []
        buy_quantity = 0
        sell_quantity = 0

        # market taking
        # lifting bids higher than fair value
        if len(order_depth.buy_orders) != 0:
            acceptable_bids = sorted([
                price for price in order_depth.buy_orders.keys() 
                if price >= fv
            ], reverse=True)

        if len(acceptable_bids) > 0:
            # we want to lift all acceptable bids
            for price in acceptable_bids:
                quantity = order_depth.buy_orders[price]
                # we want to buy the whole order position permitting
                acceptable_quantity = min(quantity, pos_lim + position - sell_quantity)
                orders.append(Order("RAINFOREST_RESIN", price, -acceptable_quantity))
                sell_quantity += acceptable_quantity
                if position - sell_quantity + pos_lim <= 0:
                    break

        # lifting asks lower than fair value
        if len(order_depth.sell_orders) != 0:
            acceptable_asks = sorted([
                price for price in order_depth.sell_orders.keys()
                if price <= fv
            ])

        if len(acceptable_asks) > 0:
            # we want to lift all acceptable asks
            for price in acceptable_asks:
                quantity = order_depth.sell_orders[price]
                # we want to buy the whole order position permitting
                acceptable_quantity = min(-quantity, pos_lim - position - buy_quantity)
                orders.append(Order("RAINFOREST_RESIN", price, acceptable_quantity))
                buy_quantity += acceptable_quantity
                if position + buy_quantity - pos_lim >= 0:
                    break

        buyVolume, sellVolume = self.clearPos(
            orders, order_depth, position, pos_lim, "RAINFOREST_RESIN", buy_quantity, sell_quantity, fv
        )
        ###
        '''
        I CHANGED THIS TO fv +- 1 but maybe we should be market making at many different levels
        '''
        ###

        buyQuant = pos_lim - (position + buyVolume)
        if buyQuant > 0:
            orders.append(Order("RAINFOREST_RESIN", round(fv + 1), buyQuant))

        sellQuant = pos_lim + (position - sellVolume)
        if sellQuant > 0:
            orders.append(Order("RAINFOREST_RESIN", round(fv - 1), -sellQuant))

    # KELP
    def kelp_fair_value(self, order_depth: OrderDepth):
        # first filter out the orders which have a high volume
        filtered_ask = [
            price
            for price in order_depth.sell_orders.keys()
            if order_depth.sell_orders[price] >=
            self.KELP["market_makers_size"]
        ]
        filtered_bid = [
            price
            for price in order_depth.buy_orders.keys()
            if order_depth.buy_orders[price] >=
            self.KELP["market_makers_size"]
        ]
        # calculated estimated fair value
        if len(filtered_ask) > 0 and len(filtered_bid) > 0:
            ask_price = min(filtered_ask)
            bid_price = max(filtered_bid)
            fair_value = (ask_price + bid_price) / 2
        # if no filtered orders, use the min and max of the order depth
        else:
            fair_value = min(order_depth.sell_orders.keys()) \
                + max(order_depth.buy_orders.keys()) / 2
        return fair_value

    def kelp_order(self,
                   order_depth: OrderDepth,
                   position: int,
                   fair_value: float) -> List[Order]:
        orders = []
        pos_lim = self.KELP["position_limit"]

        buy_quantity = 0
        sell_quantity = 0

        ###
        """
        COMPLETE THIS
        """
        ###

        pass

    # SQUID_INK
    def squid_order():
        pass

    

    def run(self, state: TradingState):
        traderObject = {}
        if state.traderData is not None and state.traderData != "":
            traderObject = jsonpickle.decode(state.traderData)

        result = {}
        conversions = 0

        # rainforest resin
        if "RAINFOREST_RESIN" in state.order_depths:
            rfr_pos = state.position["RAINFOREST_RESIN"] if "RAINFOREST_RESIN" in state.position else 0
            rfr_orders = self.rfr_orders(
                state.order_depths["RAINFOREST_RESIN"], rfr_pos
            )
            result["RAINFOREST_RESIN"] = rfr_orders

        # kelp
        if "KELP" in state.order_depths:
            kelp_fair_value = self.kelp_fair_value(state.order_depths["KELP"])
            kelp_order = self.kelp_order(
                state.order_depths["KELP"], state.position["KELP"], kelp_fair_value
            )
            result["KELP"] = kelp_order

        # squid ink
        if "SQUID_INK" in state.order_depths:
            squid_ink_order = self.squid_order(
                ###
                """
                COMPLETE THIS
                """
                ###
            )
            result["SQUID_INK"] = squid_ink_order

        # basket arb strat (no market making?)

            # basket arb of
            # pic1 and underlying
            # pic2 and underlying

        return result, conversions, traderData
