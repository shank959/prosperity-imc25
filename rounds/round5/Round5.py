from datamodel import OrderDepth, UserId, TradingState, Order, Observation, ConversionObservation  # noqa F401
from typing import List, Dict, Tuple, Any 
import string
import jsonpickle
import numpy as np
import math
from math import log, sqrt, exp
from statistics import NormalDist
import random


class Trader:
    def __init__(self):
        self.PRODUCT_HYPERPARAMS = {
            "RAINFOREST_RESIN": {
                "pos_lim": 50,
                "fair_value": 0,
            },
            "KELP": {
                "pos_lim": 50,
            },
            "SQUID_INK": {
                "pos_lim": 50,
            },
            "CROISSANTS": {
                "pos_lim": 250,
            },
            "JAMS": {
                "pos_lim": 350,
            },
            "DJEMBES": {
                "pos_lim": 60,
            },
            "PICNIC_BASKET1": {
                "pos_lim": 60,
            },
            "PICNIC_BASKET2": {
                "pos_lim": 100,
            },
            "VOLCANIC_ROCK": {
                "pos_lim": 400,
            },
            "VOLCANIC_ROCK_VOUCHER_9500": {
                "pos_lim": 300,
                "upper_threshold": 0.00322,
                "lower_threshold": -0.01665,
                "critical_boundary": 0.001,
            },
            "VOLCANIC_ROCK_VOUCHER_9750": {
                "pos_lim": 300,
                "upper_threshold": 0.03894,
                "lower_threshold": -0.00803,
                "critical_boundary": 0.002,
            },
            "VOLCANIC_ROCK_VOUCHER_10000": {
                "pos_lim": 300,
                "upper_threshold": 0.00534,
                "lower_threshold": -0.01640,
                "critical_boundary": 0.002,
            },
            "VOLCANIC_ROCK_VOUCHER_10250": {
                "pos_lim": 300,
                "upper_threshold": 0.00502,
                "lower_threshold": -0.02194,
                "critical_boundary": 0.002,
            },
            "VOLCANIC_ROCK_VOUCHER_10500": {
                "pos_lim": 300,
                "upper_threshold": float("inf"),
                "lower_threshold": float("-inf"),
                "critical_boundary": 0.0,
            },
            "MAGNIFICENT_MACARONS": {
                "pos_lim": 75
            },
        }

    # auxiliary functions

    def mid_price(self, product, order_depth: OrderDepth) -> float:
        """
        Calculate the mid price from the order depth.
        """
        best_bid = max(order_depth[product].buy_orders.keys())\
            if order_depth[product].buy_orders else 0
        best_ask = min(order_depth[product].sell_orders.keys())\
            if order_depth[product].sell_orders else 0

        return (best_bid + best_ask) / 2 if best_bid and best_ask\
            else best_bid + best_ask

    def market_maker_mid(self, product, order_depth: OrderDepth) -> float:
        """
        Calculate the market maker mid price from the order depth.
        If no market maker size known then returns midprice.
        """
        if not self.PARAMETERS[product]['mm_size']:
            return self.mid_price(product, order_depth)

        mm_bid = max((price for price, qty
                      in order_depth[product].buy_orders.items()
                      if qty > self.PARAMETERS[product]['mm_size']))\
            if order_depth[product].buy_orders else 0

        mm_ask = min((price for price, qty
                      in order_depth[product].sell_orders.items()
                      if qty > self.PARAMETERS[product]['mm_size']))\
            if order_depth[product].sell_orders else 0

        return (mm_bid + mm_ask) / 2 if mm_bid and mm_ask\
            else mm_bid + mm_ask

    # rainforest resin

    def RAINFOREST_RESIN_order(
            self,
            order_depth: OrderDepth,
            position
    ) -> List[Order]:
        """
        Create orders for the rainforest resin product.
        Buys below fair value and sells above fair value.
        Clears positions at a EV atleast 0.
        Market makes above and below fair value.
        """
        orders = []
        fair_value = self.PRODUCT_HYPERPARAMS["RAINFOREST_RESIN"]["fair_value"]
        pos_lim = self.PRODUCT_HYPERPARAMS["RAINFOREST_RESIN"]["pos_lim"]

        if order_depth["RAINFOREST_RESIN"].buy_orders:
            best_bid = max(order_depth["RAINFOREST_RESIN"].buy_orders.keys())
            if best_bid > fair_value:
                sell_amt = min(
                    order_depth["RAINFOREST_RESIN"].buy_orders[best_bid],
                    position + pos_lim
                )
                orders.append(Order("RAINFOREST_RESIN", best_bid, -sell_amt))

        if order_depth["RAINFOREST_RESIN"].sell_orders:
            best_ask = min(order_depth["RAINFOREST_RESIN"].sell_orders.keys())
            if best_ask < fair_value:
                buy_amt = min(
                    order_depth["RAINFOREST_RESIN"].sell_orders[best_ask],
                    pos_lim - position
                )
                orders.append(Order("RAINFOREST_RESIN", best_ask, buy_amt))

        return orders

    def run(
        self,
        state: TradingState,
    ) -> Tuple[Dict[str, List[Order]], int, str]:
        # initialise the output
        result = {}
        conversions = 0
        traderData = jsonpickle.decode(state.traderData)\
            if state.traderData else {}

        # get the main data
        timestamp: int = state.timestamp
        order_depth: OrderDepth = state.order_depths
        positions: Dict[str, int] = state.position

        if "RAINFOREST_RESIN" in order_depth:
            position = positions.get("RAINFOREST_RESIN", 0)
            result["RAINFOREST_RESIN"] = self.RAINFOREST_RESIN_order(
                order_depth,
                position
            )

        if "KELP" in order_depth:
            position = positions.get("KELP", 0)
            pass

        if "PICNIC_BASKET1" in order_depth:
            position = positions.get("PICNIC_BASKET1", 0)
            pass

        if "PICNIC_BASKET2" in order_depth:
            position = positions.get("PICNIC_BASKET2", 0)
            pass

        if "VOLCANIC_ROCK" in order_depth:
            position = positions.get("VOLCANIC_ROCK", 0)
            pass

        if "MAGNIFICENT_MACARONS" in order_depth:
            position = positions.get("MAGNIFICENT_MACARONS", 0)
            pass

        traderData = jsonpickle.encode(traderData)

        return result, conversions, traderData
