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
                "fair_value": 10000,
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
                "pos_lim": 75,
            },
        }

    # ============================
    # AUXILIARY FUNCTIONS SECTION
    # ============================

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

    # ============================
    # RAINFOREST_RESIN SECTION
    # ============================

    def RAINFOREST_RESIN_order(
            self,
            order_depth: OrderDepth,
            pos: int
    ) -> List[Order]:
        """
        Create orders for the rainforest resin product.
        Buys below fair value and sells above fair value.
        Clears positions at a EV atleast 0.
        Market makes above and below fair value.
        """
        orders: List[Order] = []
        fair_value = self.PRODUCT_HYPERPARAMS["RAINFOREST_RESIN"]["fair_value"]
        pos_lim = self.PRODUCT_HYPERPARAMS["RAINFOREST_RESIN"]["pos_lim"]

        print(f"Current position: {pos}")

        # === TAKE ===
        sell_amt = 0
        buy_amt = 0

        if order_depth["RAINFOREST_RESIN"].buy_orders:
            best_bid = max(order_depth["RAINFOREST_RESIN"].buy_orders.keys())
            if best_bid > fair_value:
                take_sell_amt = min(
                    order_depth["RAINFOREST_RESIN"].buy_orders[best_bid],
                    pos + pos_lim
                )
                sell_amt += take_sell_amt
                orders.append(Order(
                    "RAINFOREST_RESIN", round(best_bid), -take_sell_amt))
                print(f"[TAKE OFFER] {take_sell_amt} @ {best_bid}")

        if order_depth["RAINFOREST_RESIN"].sell_orders:
            best_ask = min(order_depth["RAINFOREST_RESIN"].sell_orders.keys())
            if best_ask < fair_value:
                take_buy_amt = min(
                    abs(order_depth["RAINFOREST_RESIN"].sell_orders[best_ask]),
                    pos_lim - pos
                )
                buy_amt += take_buy_amt
                orders.append(Order(
                    "RAINFOREST_RESIN", round(best_ask), take_buy_amt))
                print(f"[TAKE BID] {take_buy_amt} @ {best_ask}")

        pos_after_take = pos + buy_amt - sell_amt
        print(f"Position after take: {pos_after_take}")

        # === CLEAR ===
        clear_buy_amt = 0
        clear_sell_amt = 0

        if pos_after_take > 0 and\
                fair_value in order_depth["RAINFOREST_RESIN"].buy_orders:
            # want to sell
            clear_sell_amt = min(
                order_depth["RAINFOREST_RESIN"].buy_orders[fair_value],
                pos_after_take,
                pos_lim + (pos - sell_amt)
            )
            if clear_sell_amt > 0:
                sell_amt += clear_sell_amt
                orders.append(Order(
                    "RAINFOREST_RESIN", fair_value, -clear_sell_amt))
                print(f"[CLEAR OFFER] {clear_sell_amt} @ {fair_value}")

        elif pos_after_take < 0 and\
                fair_value in order_depth["RAINFOREST_RESIN"].sell_orders:
            # want to buy
            clear_buy_amt = min(
                abs(order_depth["RAINFOREST_RESIN"].sell_orders[fair_value]),
                -pos_after_take,
                pos_lim - (pos + buy_amt)
            )
            if clear_buy_amt > 0:
                buy_amt += clear_buy_amt
                orders.append(
                    Order("RAINFOREST_RESIN", fair_value, clear_buy_amt))
                print(f"[CLEAR BID] {clear_buy_amt} @ {fair_value}")

        # === MAKE ===
        asks_above_fair_value = [
            price for price in order_depth["RAINFOREST_RESIN"].sell_orders
            if price > fair_value + 1
        ]
        baaf = min(asks_above_fair_value)\
            if asks_above_fair_value else fair_value + 1
        make_sell_amt = pos_lim + (pos - sell_amt)
        print(f"Make sell amount: {make_sell_amt}")
        if make_sell_amt > 0:
            orders.append(Order("RAINFOREST_RESIN", baaf - 1, -make_sell_amt))
            print(f"[MAKE OFFER] {make_sell_amt} @ {baaf - 1}")

        bids_below_fair_value = [
            price for price in order_depth["RAINFOREST_RESIN"].buy_orders
            if price < fair_value - 1
        ]
        bbbf = max(bids_below_fair_value)\
            if bids_below_fair_value else fair_value - 1
        make_buy_amt = pos_lim - (pos + buy_amt)
        if make_buy_amt > 0:
            orders.append(Order("RAINFOREST_RESIN", bbbf + 1, make_buy_amt))
            print(f"[MAKE BID] {make_buy_amt} @ {bbbf + 1}")

        return orders

    # ============================
    # KELP SECTION
    # ============================
    def KELP_order(
            self,
            order_depth: OrderDepth,
            pos: int
    ) -> List[Order]:
        """
        Calculates fair value.
        Buys/sell below/above fairvalue -/+ width.
        Clears positions at a EV at least 0.
        Market makes above and below fair value.
        """
        orders = []
        pos_lim = self.PRODUCT_HYPERPARAMS["KELP"]["pos_lim"]

        return orders

    # ============================
    # SQUID_INK SECTION
    # ============================

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
            pos = positions.get("RAINFOREST_RESIN", 0)
            result["RAINFOREST_RESIN"] = self.RAINFOREST_RESIN_order(
                order_depth,
                pos
            )

        if "KELP" in order_depth:
            pos = positions.get("KELP", 0)
            result["KELP"] = self.KELP_order(
                order_depth,
                pos
            )

        if "PICNIC_BASKET1" in order_depth:
            pos = positions.get("PICNIC_BASKET1", 0)
            pass

        if "PICNIC_BASKET2" in order_depth:
            pos = positions.get("PICNIC_BASKET2", 0)
            pass

        if "VOLCANIC_ROCK" in order_depth:
            pos = positions.get("VOLCANIC_ROCK", 0)
            pass

        if "MAGNIFICENT_MACARONS" in order_depth:
            pos = positions.get("MAGNIFICENT_MACARONS", 0)
            pass

        traderData = jsonpickle.encode(traderData)

        return result, conversions, traderData
