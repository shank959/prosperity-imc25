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
    },
    "SQUID_INK":{
    }
}

class Trader:
    # RAINFROEST_RESIN
    def rfr_order(self, order_depth: OrderDepth, position: int) -> List[Order]:
        fv = PARAMETERS["RAINFOREST_RESIN"]["fair_value"]
        pos_lim = PARAMETERS["RAINFOREST_RESIN"]["position_limit"]

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

        

        

        
    #KELP
    def kelp_fair_value(self, order_depth: OrderDepth):
        # first filter out the orders which have a high volume
        filtered_ask = [
            price
            for price in order_depth.sell_orders.keys() 
            if order_depth.sell_orders[price] >= \
                PARAMETERS["KELP"]["market_makers_size"] 
        ]
        filtered_bid = [
            price
            for price in order_depth.buy_orders.keys() 
            if order_depth.buy_orders[price] >= \
                PARAMETERS["KELP"]["market_makers_size"] 
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
    
    def kelp_order():
        pass

    # SQUID_INK
    def squid_order():
        pass

    def run(self, state : TradingState):
        traderObject = {}
        if state.traderData != None and state.traderData != "":
            traderObject = jsonpickle.decode(state.traderData)
        
        result = {}
        conversions = 0

        # rainforest resin
        if "RAINFOREST_RESIN" in state.order_depths:
            # market take

            # clear

            # make

        # kelp
        if "KELP" in state.order_depths:
            kelp_fair_value = self.kelp_fair_value(state.order_depths["KELP"])
            # market take

            # clear

            # make

        # squid ink
        if "SQUID_INK" in state.order_depths:
            # market take

            # clear

            # make

        # basket arb strat (no market making?)

            # basket arb of
            # pic1 and underlying
            # pic2 and underlying

        return result, conversions, traderData
