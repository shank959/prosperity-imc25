from datamodel import OrderDepth, \
    TradingState, Order, Observation, ConversionObservation
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
                "take_buy_width": 1,
                "take_sell_width": 1,
                "make_buy_width": 2,
                "make_sell_width": 2,
                "reversion_beta": 1.4733,
                "mm_size": 15
            },
            "SQUID_INK": {
                "pos_lim": 50,
                "take_buy_width": 1,
                "take_sell_width": 1,
                "make_buy_width": 2,
                "make_sell_width": 2,
                "mm_size": 15,
                "reversion_beta": -1.097,
            },
            "CROISSANTS": {
                "pos_lim": 250,
                "take_buy_width": 1,
                "take_sell_width": 1,
                "make_buy_width": 2,
                "make_sell_width": 2,
            },
            "JAMS": {
                "pos_lim": 350,
                "take_buy_width": 1,
                "take_sell_width": 1,
                "make_buy_width": 2,
                "make_sell_width": 2,
            },
            "DJEMBES": {
                "pos_lim": 60,
                "take_buy_width": 1,
                "take_sell_width": 1,
                "make_buy_width": 2,
                "make_sell_width": 2,
            },
            "PICNIC_BASKET1": {
                "pos_lim": 60,
                "take_buy_width": 10,
                "take_sell_width": 10,
                "make_buy_width": 2,
                "make_sell_width": 2,
                "premium": -152.22,
            },
            "PICNIC_BASKET2": {
                "pos_lim": 100,
                "take_buy_width": 16,
                "take_sell_width": 16,
                "make_buy_width": 2,
                "make_sell_width": 2,
                "premium": 66.108,
            },
            "VOLCANIC_ROCK": {
                "pos_lim": 400,
                "strikes": [
                    9500, 9750, 10000, 10250, 10500
                ]
            },
            "VOLCANIC_ROCK_VOUCHER_9500": {
                "pos_lim": 200,
                "take_buy_width": 1,
                "take_sell_width": 1,
                "make_buy_width": 2,
                "make_sell_width": 2,
                "clear_width": 1,
                "upper_threshold": 0.00322,
                "lower_threshold": -0.01665,
                "critical_boundary": 0.001,
            },
            "VOLCANIC_ROCK_VOUCHER_9750": {
                "pos_lim": 200,
                "take_buy_width": 1,
                "take_sell_width": 1,
                "make_buy_width": 2,
                "make_sell_width": 2,
                "clear_width": 1,
                "upper_threshold": 0.03894,
                "lower_threshold": -0.00803,
                "critical_boundary": 0.002,
            },
            "VOLCANIC_ROCK_VOUCHER_10000": {
                "pos_lim": 200,
                "take_buy_width": 1,
                "take_sell_width": 1,
                "make_buy_width": 1,
                "make_sell_width": 1,
                "clear_width": 1,
                "upper_threshold": 0.00534,
                "lower_threshold": -0.01640,
                "critical_boundary": 0.002,
            },
            "VOLCANIC_ROCK_VOUCHER_10250": {
                "pos_lim": 200,
                "take_buy_width": 1,
                "take_sell_width": 1,
                "make_buy_width": 2,
                "make_sell_width": 2,
                "upper_threshold": 0.00502,
                "lower_threshold": -0.02194,
                "critical_boundary": 0.002,
            },
            "VOLCANIC_ROCK_VOUCHER_10500": {
                "pos_lim": 200,
                "take_buy_width": 1,
                "take_sell_width": 1,
                "make_buy_width": 2,
                "make_sell_width": 2,
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
        If no market maker size is known, then returns the mid price.
        """
        if 'mm_size' not in self.PRODUCT_HYPERPARAMS[product]:
            return self.mid_price(product, order_depth)

        mm_size = self.PRODUCT_HYPERPARAMS[product]['mm_size']

        mm_bid = max((price for price, qty
                      in order_depth[product].buy_orders.items()
                      if qty >= mm_size), default=0)

        mm_ask = min((price for price, qty
                      in order_depth[product].sell_orders.items()
                      if abs(qty) >= mm_size), default=0)

        return (mm_bid + mm_ask) / 2 if mm_bid and mm_ask\
            else self.mid_price(product, order_depth)

    def general_order_machine(
        self,
        order_depth: OrderDepth,
        product: str,
        pos: int,
    ) -> List[Order]:
        """
        General order machine for the products
        which does market taking, clearing and making.
        """
        orders = []
        pos_lim = self.PRODUCT_HYPERPARAMS[product]["pos_lim"]
        take_buy_width = self.PRODUCT_HYPERPARAMS[product].get(
            "take_buy_width", 1)
        take_sell_width = self.PRODUCT_HYPERPARAMS[product].get(
            "take_sell_width", 1)
        make_buy_width = self.PRODUCT_HYPERPARAMS[product].get(
            "make_buy_width", 1)
        make_sell_width = self.PRODUCT_HYPERPARAMS[product].get(
            "make_sell_width", 1)
        fair_value = self.market_maker_mid(product, order_depth)

        # === TAKE ===
        sell_amt = 0
        buy_amt = 0

        if order_depth[product].buy_orders:
            best_bid = max(order_depth[product].buy_orders.keys())
            if best_bid > fair_value + take_sell_width:
                take_sell_amt = min(
                    order_depth[product].buy_orders[best_bid],
                    pos + pos_lim
                )
                sell_amt += take_sell_amt
                orders.append(Order(
                    product, round(best_bid), -take_sell_amt))
                print(f"[TAKE OFFER] {take_sell_amt} @ {best_bid}")

        if order_depth[product].sell_orders:
            best_ask = min(order_depth[product].sell_orders.keys())
            if best_ask < fair_value - take_buy_width:
                take_buy_amt = min(
                    abs(order_depth[product].sell_orders[best_ask]),
                    pos_lim - pos
                )
                buy_amt += take_buy_amt
                orders.append(Order(
                    product, round(best_ask), take_buy_amt))
                print(f"[TAKE BID] {take_buy_amt} @ {best_ask}")

        pos_after_take = pos + buy_amt - sell_amt
        print(f"Position after take: {pos_after_take}")

        # === CLEAR ===
        clear_buy_amt = 0
        clear_sell_amt = 0
        fair_ask = math.ceil(fair_value)
        fair_bid = math.floor(fair_value)

        if pos_after_take > 0 and\
                fair_ask in order_depth[product].buy_orders:
            # want to sell
            clear_sell_amt = min(
                order_depth[product].buy_orders[fair_ask],
                pos_after_take,
                pos_lim + (pos - sell_amt)
            )
            if clear_sell_amt > 0:
                sell_amt += clear_sell_amt
                orders.append(Order(
                    product, fair_ask, -clear_sell_amt))
                print(f"[CLEAR OFFER] {clear_sell_amt} @ {fair_ask}")

        elif pos_after_take < 0 and\
                fair_bid in order_depth[product].sell_orders:
            # want to buy
            clear_buy_amt = min(
                abs(order_depth[product].sell_orders[fair_bid]),
                -pos_after_take,
                pos_lim - (pos + buy_amt)
            )
            if clear_buy_amt > 0:
                buy_amt += clear_buy_amt
                orders.append(
                    Order(product, fair_bid, clear_buy_amt))
                print(f"[CLEAR BID] {clear_buy_amt} @ {fair_bid}")

        # === MAKE ===
        asks_above_fair_value = [
            price for price in order_depth[product].sell_orders
            if price > fair_value + make_sell_width
        ]
        baaf = min(asks_above_fair_value)\
            if asks_above_fair_value\
            else math.ceil(fair_value + make_sell_width)
        make_sell_amt = pos_lim + (pos - sell_amt)
        print(f"Make sell amount: {make_sell_amt}")
        if make_sell_amt > 0:
            orders.append(Order(product, baaf - 1, -make_sell_amt))
            print(f"[MAKE OFFER] {make_sell_amt} @ {baaf - 1}")

        bids_below_fair_value = [
            price for price in order_depth[product].buy_orders
            if price < fair_value - make_buy_width
        ]
        bbbf = max(bids_below_fair_value)\
            if bids_below_fair_value\
            else math.ceil(fair_value - make_buy_width)
        make_buy_amt = pos_lim - (pos + buy_amt)
        if make_buy_amt > 0:
            orders.append(Order(product, bbbf + 1, make_buy_amt))
            print(f"[MAKE BID] {make_buy_amt} @ {bbbf + 1}")

        return orders

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
            pos: int,
            historical: List[float]
    ) -> List[Order]:
        """
        Calculates fair value.
        Buys/sell below/above fairvalue -/+ width.
        Clears positions at a EV at least 0.
        Market makes above and below fair value.
        """
        orders = []
        pos_lim = self.PRODUCT_HYPERPARAMS["KELP"]["pos_lim"]
        take_buy_width = self.PRODUCT_HYPERPARAMS["KELP"].get(
            "take_buy_width", 1)
        take_sell_width = self.PRODUCT_HYPERPARAMS["KELP"].get(
            "take_sell_width", 1)
        make_buy_width = self.PRODUCT_HYPERPARAMS["KELP"].get(
            "make_buy_width", 1)
        make_sell_width = self.PRODUCT_HYPERPARAMS["KELP"].get(
            "make_sell_width", 1)

        # === CALCULATE FAIR VALUE === # TODO IMPROVE
        # mm_mid = self.market_maker_mid("KELP", order_depth)
        # reversion_beta = self.PRODUCT_HYPERPARAMS["KELP"].get(
        #     "reversion_beta", 1)
        # if historical:
        #     prev_mm_mid = historical[-1]
        #     fair_value = (
        #         ((mm_mid - prev_mm_mid)/prev_mm_mid
        #             * reversion_beta) * mm_mid + mm_mid
        #     )
        # else:
        #     fair_value = mm_mid

        fair_value = self.market_maker_mid("KELP", order_depth)
        # historical.append(fair_value)

        # === TAKE ===
        sell_amt = 0
        buy_amt = 0

        if order_depth["KELP"].buy_orders:
            best_bid = max(order_depth["KELP"].buy_orders.keys())
            if best_bid > fair_value + take_sell_width:
                take_sell_amt = min(
                    order_depth["KELP"].buy_orders[best_bid],
                    pos + pos_lim
                )
                sell_amt += take_sell_amt
                orders.append(Order(
                    "KELP", round(best_bid), -take_sell_amt))
                print(f"[TAKE OFFER] {take_sell_amt} @ {best_bid}")

        if order_depth["KELP"].sell_orders:
            best_ask = min(order_depth["KELP"].sell_orders.keys())
            if best_ask < fair_value - take_buy_width:
                take_buy_amt = min(
                    abs(order_depth["KELP"].sell_orders[best_ask]),
                    pos_lim - pos
                )
                buy_amt += take_buy_amt
                orders.append(Order(
                    "KELP", round(best_ask), take_buy_amt))
                print(f"[TAKE BID] {take_buy_amt} @ {best_ask}")

        pos_after_take = pos + buy_amt - sell_amt
        print(f"Position after take: {pos_after_take}")

        # === CLEAR ===
        clear_buy_amt = 0
        clear_sell_amt = 0
        fair_ask = math.ceil(fair_value)
        fair_bid = math.floor(fair_value)

        if pos_after_take > 0 and\
                fair_ask in order_depth["KELP"].buy_orders:
            # want to sell
            clear_sell_amt = min(
                order_depth["KELP"].buy_orders[fair_ask],
                pos_after_take,
                pos_lim + (pos - sell_amt)
            )
            if clear_sell_amt > 0:
                sell_amt += clear_sell_amt
                orders.append(Order(
                    "KELP", fair_ask, -clear_sell_amt))
                print(f"[CLEAR OFFER] {clear_sell_amt} @ {fair_ask}")

        elif pos_after_take < 0 and\
                fair_bid in order_depth["KELP"].sell_orders:
            # want to buy
            clear_buy_amt = min(
                abs(order_depth["KELP"].sell_orders[fair_bid]),
                -pos_after_take,
                pos_lim - (pos + buy_amt)
            )
            if clear_buy_amt > 0:
                buy_amt += clear_buy_amt
                orders.append(
                    Order("KELP", fair_bid, clear_buy_amt))
                print(f"[CLEAR BID] {clear_buy_amt} @ {fair_bid}")

        # === MAKE ===
        asks_above_fair_value = [
            price for price in order_depth["KELP"].sell_orders
            if price > fair_value + make_sell_width
        ]
        baaf = min(asks_above_fair_value)\
            if asks_above_fair_value\
            else math.ceil(fair_value + make_sell_width)
        make_sell_amt = pos_lim + (pos - sell_amt)
        print(f"Make sell amount: {make_sell_amt}")
        if make_sell_amt > 0:
            orders.append(Order("KELP", baaf - 1, -make_sell_amt))
            print(f"[MAKE OFFER] {make_sell_amt} @ {baaf - 1}")

        bids_below_fair_value = [
            price for price in order_depth["KELP"].buy_orders
            if price < fair_value - make_buy_width
        ]
        bbbf = max(bids_below_fair_value)\
            if bids_below_fair_value\
            else math.ceil(fair_value - make_buy_width)
        make_buy_amt = pos_lim - (pos + buy_amt)
        if make_buy_amt > 0:
            orders.append(Order("KELP", bbbf + 1, make_buy_amt))
            print(f"[MAKE BID] {make_buy_amt} @ {bbbf + 1}")

        return orders, historical

    # ============================
    # SQUID_INK SECTION
    # ============================

    def SQUID_INK_order(
            self,
            order_depth: OrderDepth,
            pos: int,
            historical: List[float]
    ) -> List[Order]:
        """
        Calculates fair value using mean reversion beta.
        Buys/sell below/above fairvalue -/+ width.
        Clears positions at a EV at least 0.
        Market makes above and below fair value.
        """
        orders = []
        pos_lim = self.PRODUCT_HYPERPARAMS["SQUID_INK"]["pos_lim"]
        take_buy_width = self.PRODUCT_HYPERPARAMS["SQUID_INK"].get(
            "take_buy_width", 1)
        take_sell_width = self.PRODUCT_HYPERPARAMS["SQUID_INK"].get(
            "take_sell_width", 1)
        make_buy_width = self.PRODUCT_HYPERPARAMS["SQUID_INK"].get(
            "make_buy_width", 1)
        make_sell_width = self.PRODUCT_HYPERPARAMS["SQUID_INK"].get(
            "make_sell_width", 1)

        # === CALCULATE FAIR VALUE ===
        mm_mid = self.market_maker_mid("SQUID_INK", order_depth)
        reversion_beta = self.PRODUCT_HYPERPARAMS["SQUID_INK"].get(
            "reversion_beta", 1)
        if historical:
            prev_mm_mid = historical[-1]
            fair_value = (
                ((mm_mid - prev_mm_mid)/prev_mm_mid
                    * reversion_beta) * mm_mid + mm_mid
            )
        else:
            fair_value = mm_mid

        # historical.append(fair_value)

        # === TAKE ===
        sell_amt = 0
        buy_amt = 0

        if order_depth["SQUID_INK"].buy_orders:
            best_bid = max(order_depth["SQUID_INK"].buy_orders.keys())
            if best_bid > fair_value + take_sell_width:
                take_sell_amt = min(
                    order_depth["SQUID_INK"].buy_orders[best_bid],
                    pos + pos_lim
                )
                sell_amt += take_sell_amt
                orders.append(Order(
                    "SQUID_INK", round(best_bid), -take_sell_amt))
                print(f"[TAKE OFFER] {take_sell_amt} @ {best_bid}")

        if order_depth["SQUID_INK"].sell_orders:
            best_ask = min(order_depth["SQUID_INK"].sell_orders.keys())
            if best_ask < fair_value - take_buy_width:
                take_buy_amt = min(
                    abs(order_depth["SQUID_INK"].sell_orders[best_ask]),
                    pos_lim - pos
                )
                buy_amt += take_buy_amt
                orders.append(Order(
                    "SQUID_INK", round(best_ask), take_buy_amt))
                print(f"[TAKE BID] {take_buy_amt} @ {best_ask}")

        pos_after_take = pos + buy_amt - sell_amt
        print(f"Position after take: {pos_after_take}")

        # === CLEAR ===
        clear_buy_amt = 0
        clear_sell_amt = 0
        fair_ask = math.ceil(fair_value)
        fair_bid = math.floor(fair_value)

        if pos_after_take > 0 and\
                fair_ask in order_depth["SQUID_INK"].buy_orders:
            # want to sell
            clear_sell_amt = min(
                order_depth["SQUID_INK"].buy_orders[fair_ask],
                pos_after_take,
                pos_lim + (pos - sell_amt)
            )
            if clear_sell_amt > 0:
                sell_amt += clear_sell_amt
                orders.append(Order(
                    "SQUID_INK", fair_ask, -clear_sell_amt))
                print(f"[CLEAR OFFER] {clear_sell_amt} @ {fair_ask}")

        elif pos_after_take < 0 and\
                fair_bid in order_depth["SQUID_INK"].sell_orders:
            # want to buy
            clear_buy_amt = min(
                abs(order_depth["SQUID_INK"].sell_orders[fair_bid]),
                -pos_after_take,
                pos_lim - (pos + buy_amt)
            )
            if clear_buy_amt > 0:
                buy_amt += clear_buy_amt
                orders.append(
                    Order("SQUID_INK", fair_bid, clear_buy_amt))
                print(f"[CLEAR BID] {clear_buy_amt} @ {fair_bid}")

        # === MAKE ===
        asks_above_fair_value = [
            price for price in order_depth["SQUID_INK"].sell_orders
            if price > fair_value + make_sell_width
        ]
        baaf = min(asks_above_fair_value)\
            if asks_above_fair_value\
            else math.ceil(fair_value + make_sell_width)
        make_sell_amt = pos_lim + (pos - sell_amt)
        print(f"Make sell amount: {make_sell_amt}")
        if make_sell_amt > 0:
            orders.append(Order(
                "SQUID_INK", baaf - 1, -make_sell_amt))
            print(f"[MAKE OFFER] {make_sell_amt} @ {baaf - 1}")

        bids_below_fair_value = [
            price for price in order_depth["SQUID_INK"].buy_orders
            if price < fair_value - make_buy_width
        ]
        bbbf = max(bids_below_fair_value)\
            if bids_below_fair_value\
            else math.floor(fair_value - make_buy_width)
        make_buy_amt = pos_lim - (pos + buy_amt)
        if make_buy_amt > 0:
            orders.append(Order(
                "SQUID_INK", bbbf + 1, make_buy_amt))
            print(f"[MAKE BID] {make_buy_amt} @ {bbbf + 1}")

        return orders, historical

    # ============================
    # PICNIC_BASKET 1 SECTION
    # ============================

    def PICNIC_BASKET1_order(
            self,
            order_depth: OrderDepth,
            pos: int
    ) -> List[Order]:
        """
        Calculates fair value based on the underlying
        """
        orders = []
        pos_lim = self.PRODUCT_HYPERPARAMS["PICNIC_BASKET1"]["pos_lim"]
        take_buy_width = self.PRODUCT_HYPERPARAMS["PICNIC_BASKET1"].get(
            "take_buy_width", 1)
        take_sell_width = self.PRODUCT_HYPERPARAMS["PICNIC_BASKET1"].get(
            "take_sell_width", 1)
        make_buy_width = self.PRODUCT_HYPERPARAMS["PICNIC_BASKET1"].get(
            "make_buy_width", 1)
        make_sell_width = self.PRODUCT_HYPERPARAMS["PICNIC_BASKET1"].get(
            "make_sell_width", 1)
        premium = self.PRODUCT_HYPERPARAMS["PICNIC_BASKET1"].get(
            "premium", 0)

        # === CALCULATE FAIR VALUE ===
        CROISSANTS = self.mid_price("CROISSANTS", order_depth)
        JAMS = self.mid_price("JAMS", order_depth)
        DJEMBES = self.mid_price("DJEMBES", order_depth)
        fair_value = 6 * CROISSANTS + 3 * JAMS + DJEMBES + premium

        print(f"Fair value: {fair_value}")

        # === TAKE ===
        sell_amt = 0
        buy_amt = 0
        if order_depth["PICNIC_BASKET1"].buy_orders:
            best_bid = max(order_depth["PICNIC_BASKET1"].buy_orders.keys())
            print(f"Best bid: {best_bid}")
            if best_bid > fair_value + take_sell_width:
                take_sell_amt = min(
                    order_depth["PICNIC_BASKET1"].buy_orders[best_bid],
                    pos + pos_lim
                )
                sell_amt += take_sell_amt
                orders.append(Order(
                    "PICNIC_BASKET1", round(best_bid), -take_sell_amt))
                print(f"[TAKE OFFER] {take_sell_amt} @ {best_bid}")

        if order_depth["PICNIC_BASKET1"].sell_orders:
            best_ask = min(order_depth["PICNIC_BASKET1"].sell_orders.keys())
            print(f"Best ask: {best_ask}")
            if best_ask < fair_value - take_buy_width:
                take_buy_amt = min(
                    abs(order_depth["PICNIC_BASKET1"].sell_orders[best_ask]),
                    pos_lim - pos
                )
                buy_amt += take_buy_amt
                orders.append(Order(
                    "PICNIC_BASKET1", round(best_ask), take_buy_amt))
                print(f"[TAKE BID] {take_buy_amt} @ {best_ask}")

        pos_after_take = pos + buy_amt - sell_amt
        print(f"Position after take: {pos_after_take}")

        return orders

    # ============================
    # PICNIC_BASKET 2 SECTION
    # ============================

    def PICNIC_BASKET2_order(
            self,
            order_depth: OrderDepth,
            pos: int
    ) -> List[Order]:
        """
        Calculates fair value based on the underlying
        """
        orders = []
        pos_lim = self.PRODUCT_HYPERPARAMS["PICNIC_BASKET2"]["pos_lim"]
        take_buy_width = self.PRODUCT_HYPERPARAMS["PICNIC_BASKET2"].get(
            "take_buy_width", 1)
        take_sell_width = self.PRODUCT_HYPERPARAMS["PICNIC_BASKET2"].get(
            "take_sell_width", 1)
        make_buy_width = self.PRODUCT_HYPERPARAMS["PICNIC_BASKET2"].get(
            "make_buy_width", 1)
        make_sell_width = self.PRODUCT_HYPERPARAMS["PICNIC_BASKET2"].get(
            "make_sell_width", 1)
        premium = self.PRODUCT_HYPERPARAMS["PICNIC_BASKET2"].get(
            "premium", 0)

        # === CALCULATE FAIR VALUE ===
        CROISSANTS = self.mid_price("CROISSANTS", order_depth)
        JAMS = self.mid_price("JAMS", order_depth)
        fair_value = 4 * CROISSANTS + 2 * JAMS + premium

        # === TAKE ===
        sell_amt = 0
        buy_amt = 0
        if order_depth["PICNIC_BASKET2"].buy_orders:
            best_bid = max(order_depth["PICNIC_BASKET2"].buy_orders.keys())
            if best_bid > fair_value + take_sell_width:
                take_sell_amt = min(
                    order_depth["PICNIC_BASKET2"].buy_orders[best_bid],
                    pos + pos_lim
                )
                sell_amt += take_sell_amt
                orders.append(Order(
                    "PICNIC_BASKET2", round(best_bid), -take_sell_amt))
                print(f"[TAKE OFFER] {take_sell_amt} @ {best_bid}")

        if order_depth["PICNIC_BASKET2"].sell_orders:
            best_ask = min(order_depth["PICNIC_BASKET2"].sell_orders.keys())
            if best_ask < fair_value - take_buy_width:
                take_buy_amt = min(
                    abs(order_depth["PICNIC_BASKET2"].sell_orders[best_ask]),
                    pos_lim - pos
                )
                buy_amt += take_buy_amt
                orders.append(Order(
                    "PICNIC_BASKET2", round(best_ask), take_buy_amt))
                print(f"[TAKE BID] {take_buy_amt} @ {best_ask}")

        pos_after_take = pos + buy_amt - sell_amt
        print(f"Position after take: {pos_after_take}")

        return orders

    # ============================
    # DJEMBES SECTION
    # ============================
    def SYNTH_DJEMBES_order(
            self,
            order_depth: OrderDepth,
            positions: Dict[str, int]
    ) -> List[Order]:
        """
        Synthetic DJEMBES order strategy.
        """
        orders = []
        p1_pos_lim = self.PRODUCT_HYPERPARAMS["PICNIC_BASKET1"]["pos_lim"]
        p2_pos_lim = self.PRODUCT_HYPERPARAMS["PICNIC_BASKET2"]["pos_lim"]

        p1_pos = p1_pos_after_take = positions.get("PICNIC_BASKET1", 0)
        p2_pos = p2_pos_after_take = positions.get("PICNIC_BASKET2", 0)

        fair_width = 20

        d_mid = self.mid_price("DJEMBES", order_depth)
        print(f"DJEMBES mid price: {d_mid}")
        tp1m3p2_fair_value = 2 * d_mid - 502
        print(f"TP1M3P2 fair value: {tp1m3p2_fair_value}")
        print(f"Current position: {p1_pos} {p2_pos}")

        # === TAKE ===
        if order_depth["PICNIC_BASKET1"].buy_orders\
                and order_depth["PICNIC_BASKET2"].sell_orders:
            best_p1_bid = max(order_depth["PICNIC_BASKET1"].buy_orders.keys())
            best_p2_ask = min(order_depth["PICNIC_BASKET2"].sell_orders.keys())
            best_synthetic_bid = 2 * best_p1_bid - 3 * best_p2_ask
            print(f"Best synthetic bid: {best_synthetic_bid}")
            if best_synthetic_bid > tp1m3p2_fair_value + fair_width:
                take_sell_amt = min(
                    order_depth["PICNIC_BASKET1"].buy_orders[best_p1_bid] // 2,
                    abs(
                        order_depth["PICNIC_BASKET2"].sell_orders[best_p2_ask]
                    ) // 3,
                    (p1_pos_lim + p1_pos) // 2,
                    (p2_pos_lim - p2_pos) // 3
                )
                print(f"Take sell amount: {take_sell_amt}")
                if take_sell_amt > 0:
                    orders.extend([
                        Order("PICNIC_BASKET1",
                              best_p1_bid, - 2 * take_sell_amt),
                        Order("PICNIC_BASKET2",
                              best_p2_ask, 3 * take_sell_amt)
                    ])
                    p1_pos_after_take -= take_sell_amt * 2
                    p2_pos_after_take += take_sell_amt * 3
                    print(
                        f"[TAKE OFFER] {take_sell_amt} @ {best_synthetic_bid}")

        if order_depth["PICNIC_BASKET1"].sell_orders\
                and order_depth["PICNIC_BASKET2"].buy_orders:
            best_p1_ask = min(order_depth["PICNIC_BASKET1"].sell_orders.keys())
            best_p2_bid = max(order_depth["PICNIC_BASKET2"].buy_orders.keys())
            best_synthetic_ask = 2 * best_p1_ask - 3 * best_p2_bid
            print(f"Best synthetic ask: {best_synthetic_ask}")
            if best_synthetic_ask < tp1m3p2_fair_value - fair_width:
                take_buy_amt = min(
                    abs(
                        order_depth["PICNIC_BASKET1"].sell_orders[best_p1_ask]
                    ) // 2,
                    order_depth["PICNIC_BASKET2"].buy_orders[best_p2_bid] // 3,
                    (p1_pos_lim - (p1_pos_after_take)) // 2,
                    (p2_pos_lim + p2_pos_after_take) // 3
                )
                print(f"Take buy amount: {take_buy_amt}")
                if take_buy_amt > 0:
                    orders.extend([
                        Order("PICNIC_BASKET1",
                              round(best_p1_ask), 2 * take_buy_amt),
                        Order("PICNIC_BASKET2",
                              round(best_p2_bid), -3 * take_buy_amt)
                    ])
                    p1_pos_after_take += take_buy_amt * 2
                    p2_pos_after_take -= take_buy_amt * 3
                    print(
                        f"[TAKE BID] {take_buy_amt} @ {best_synthetic_ask}")

        # === CLEAR ===
        clear_buy_amt = 0
        clear_sell_amt = 0
        fair_ask = math.ceil(tp1m3p2_fair_value)
        fair_bid = math.floor(tp1m3p2_fair_value)

        return orders

    # ============================
    # VOLCANIC_ROCK SECTION
    # ============================

    def bisection_iv(
            self,
            S: float,
            V: float,
            K: int,
            TTE: float,
            low: float = 1e-4,
            high: float = 1.0,
            tol: float = 1e-4,
            max_iter: int = 10
    ) -> Tuple[float, float]:
        """
        Bisection method to find the implied volatility and delta.
        """
        N = NormalDist()
        for _ in range(max_iter):
            iv = (low + high) / 2
            d1 = (log(S / K) + (0.5 * high**2) * TTE) / (high * sqrt(TTE))
            d2 = d1 - high * sqrt(TTE)
            delta = N.cdf(d1)
            price = S * delta - K * N.cdf(d2)
            if abs(price - V) < tol:
                return iv, delta
            if price > V:
                high = iv
            else:
                low = iv
        return iv, delta

    def newton_iv(
            self,
            S: float,
            V: float,
            K: int,
            TTE: float,
            iv: float = 0.2,
            tol: float = 1e-4,
            max_iter: int = 50
    ) -> Tuple[float, float]:
        """
        Newton's method to find the implied volatility and delta.
        """
        N = NormalDist()
        for _ in range(max_iter):
            d1 = (log(S / K) + (0.5 * iv**2) * TTE) / (iv * sqrt(TTE))
            d2 = d1 - iv * sqrt(TTE)
            delta = N.cdf(d1)
            price = S * delta - K * N.cdf(d2)
            vega = S * N.pdf(d1) * sqrt(TTE)
            if vega < 1e-5:
                return self.bisection_iv(
                    S, V, K, TTE, low=1e-4, high=1.0, tol=tol)
            iv -= (price - V) / vega
            if abs(price - V) < tol:
                return iv, delta
        return iv, delta

    def black_scholes_price(
            self,
            S: float,
            K: int,
            TTE: float,
            iv: float,
            r: float = 0.0
    ) -> float:
        """
        Calculate the Black-Scholes price.
        """
        N = NormalDist()
        d1 = (log(S / K) + (0.5 * iv**2) * TTE) / (iv * sqrt(TTE))
        d2 = d1 - iv * sqrt(TTE)
        call_price = S * N.cdf(d1) - K * exp(-r * TTE) * N.cdf(d2)
        return call_price

    def VOLCANIC_ROCK_order(
            self,
            order_depth: OrderDepth,
            positions: Dict[str, int],
            timestamp: int
    ) -> List[Order]:
        """
        """
        orders = []
        strikes = self.PRODUCT_HYPERPARAMS["VOLCANIC_ROCK"]["strikes"]

        TTE = (3 * 1e6 - timestamp)/(1e6 * 365)
        S = self.mid_price("VOLCANIC_ROCK", order_depth)

        # === CALCULATE IV ===
        info = {}
        for strike in strikes:
            product = f"VOLCANIC_ROCK_VOUCHER_{strike}"
            V = self.mid_price(product, order_depth)
            iv, delta = self.newton_iv(
                S,
                V,
                strike,
                TTE
            )
            info[product] = {
                "strike": strike,
                "mid_price": V,
                "iv": iv,
                "delta": delta,
                "moneyness": np.log(strike/S)/sqrt(TTE),
            }

        # === FIT IV CURVE ===
        coeffs = np.polyfit(
            [info[product]["moneyness"] for product in info],
            [info[product]["iv"] for product in info],
            2
        )

        for product in info:
            info[product]["iv_fit"] =\
                coeffs[0] * info[product]["moneyness"]**2\
                + coeffs[1] * info[product]["moneyness"]\
                + coeffs[2]

        for product in info:
            info[product]["misvol"] =\
                (info[product]["iv"] - info[product]["iv_fit"])

        # === COMPUTE FAIR PRICE ===
        for product, data in info.items():
            if data['iv_fit'] <= 0:
                data["fair_value"] = 0.0
                continue
            data["fair_value"] = self.black_scholes_price(
                S,
                data["strike"],
                TTE,
                data["iv_fit"]
            )

        print(info)

        # === ORDERS ===
        pos_lim =\
            self.PRODUCT_HYPERPARAMS["VOLCANIC_ROCK_VOUCHER_10000"]["pos_lim"]
        orders = []

        for product, data in [("VOLCANIC_ROCK_VOUCHER_10000",
                               info["VOLCANIC_ROCK_VOUCHER_10000"])]:
            pos = positions.get(product, 0)
            take_buy_width = self.PRODUCT_HYPERPARAMS[product].get(
                "take_buy_width", 1)
            take_sell_width = self.PRODUCT_HYPERPARAMS[product].get(
                "take_sell_width", 1)
            make_buy_width = self.PRODUCT_HYPERPARAMS[product].get(
                "make_buy_width", 1)
            make_sell_width = self.PRODUCT_HYPERPARAMS[product].get(
                "make_sell_width", 1)
            clear_width = self.PRODUCT_HYPERPARAMS[product].get(
                "clear_width", 1)
            fair_value = data["fair_value"]

            # === TAKE ===
            sell_amt = 0
            buy_amt = 0

            if order_depth[product].buy_orders:
                best_bid = max(order_depth[product].buy_orders.keys())
                if best_bid > fair_value + take_sell_width:
                    take_sell_amt = min(
                        order_depth[product].buy_orders[best_bid],
                        pos + pos_lim
                    )
                    sell_amt += take_sell_amt
                    orders.append(Order(
                        product, round(best_bid), -take_sell_amt))
                    print(f"[TAKE OFFER] {take_sell_amt} @ {best_bid}")

            if order_depth[product].sell_orders:
                best_ask = min(order_depth[product].sell_orders.keys())
                if best_ask < fair_value - take_buy_width:
                    take_buy_amt = min(
                        abs(order_depth[product].sell_orders[best_ask]),
                        pos_lim - pos
                    )
                    buy_amt += take_buy_amt
                    orders.append(Order(
                        product, round(best_ask), take_buy_amt))
                    print(f"[TAKE BID] {take_buy_amt} @ {best_ask}")

            pos_after_take = pos + buy_amt - sell_amt
            print(f"Position after take: {pos_after_take}")

            # === CLEAR ===
            clear_buy_amt = 0
            clear_sell_amt = 0
            fair_ask = math.ceil(fair_value)
            fair_bid = math.floor(fair_value)

            if pos_after_take > 0:
                # want to sell
                better_asks = [
                    price for price in order_depth[product].buy_orders
                    if price >= fair_ask - clear_width
                ]
                if better_asks:
                    best_ask = max(better_asks)
                    clear_sell_amt = min(
                        order_depth[product].buy_orders[best_ask],
                        pos_after_take,
                        pos_lim + (pos - sell_amt)
                    )
                    if clear_sell_amt > 0:
                        sell_amt += clear_sell_amt
                        orders.append(Order(
                            product, best_ask, -clear_sell_amt))
                        print(f"[CLEAR OFFER] {clear_sell_amt} @ {best_ask}")

            elif pos_after_take < 0:
                # want to buy
                better_bids = [
                    price for price in order_depth[product].sell_orders
                    if price <= fair_bid + clear_width
                ]
                if better_bids:
                    best_bid = min(better_bids)
                    clear_buy_amt = min(
                        abs(order_depth[product].sell_orders[best_bid]),
                        -pos_after_take,
                        pos_lim - (pos + buy_amt)
                    )
                    if clear_buy_amt > 0:
                        buy_amt += clear_buy_amt
                        orders.append(
                            Order(product, best_bid, clear_buy_amt))
                        print(f"[CLEAR BID] {clear_buy_amt} @ {best_bid}")

            # === MAKE ===
            asks_above_fair_value = [
                price for price in order_depth[product].sell_orders
                if price > fair_value + make_sell_width
            ]
            baaf = min(asks_above_fair_value)\
                if asks_above_fair_value\
                else math.ceil(fair_value + make_sell_width)
            make_sell_amt = min(pos_lim + (pos - sell_amt), 20)
            print(f"Make sell amount: {make_sell_amt}")
            if make_sell_amt > 0:
                orders.append(Order(
                    product, baaf - 1, -make_sell_amt))
                print(f"[MAKE OFFER] {make_sell_amt} @ {baaf - 1}")

            bids_below_fair_value = [
                price for price in order_depth[product].buy_orders
                if price < fair_value - make_buy_width
            ]
            bbbf = max(bids_below_fair_value)\
                if bids_below_fair_value\
                else math.floor(fair_value - make_buy_width)
            make_buy_amt = min(pos_lim - (pos + buy_amt), 20)
            if make_buy_amt > 0:
                orders.append(Order(
                    product, bbbf + 1, make_buy_amt))
                print(f"[MAKE BID] {make_buy_amt} @ {bbbf + 1}")

        return orders

    # ============================
    # MAIN
    # ============================

    def run(
        self,
        state: TradingState,
    ) -> Tuple[Dict[str, List[Order]], int, str]:
        # initialise the output
        result = {}
        conversions = 0

        # === STATE DATA ===
        timestamp: int = state.timestamp
        order_depth: OrderDepth = state.order_depths
        positions: Dict[str, int] = state.position

        # === TRADER DATA ===
        traderData = jsonpickle.decode(state.traderData)\
            if state.traderData else {}
        kelp_historical = traderData.get("kelp_historical", [])
        squid_ink_historical = traderData.get("squid_ink_historical", [])

        # === ORDER CALLS ===
        if "RAINFOREST_RESIN" in order_depth:
            pos = positions.get("RAINFOREST_RESIN", 0)
            result["RAINFOREST_RESIN"] = self.RAINFOREST_RESIN_order(
                order_depth,
                pos
            )

        if "KELP" in order_depth:
            pos = positions.get("KELP", 0)
            result["KELP"], kelp_historical = self.KELP_order(
                order_depth,
                pos,
                kelp_historical
            )

        if "SQUID_INK" in order_depth:
            pos = positions.get("SQUID_INK", 0)
            result["SQUID_INK"], squid_ink_historical = self.SQUID_INK_order(
                order_depth,
                pos,
                squid_ink_historical
            )

        # if "CROISSANTS" in order_depth:
        #     pos = positions.get("CROISSANTS", 0)
        #     orders = self.general_order_machine(
        #         order_depth,
        #         "CROISSANTS",
        #         pos,
        #     )
        #     result["CROISSANTS"] = orders

        # if "JAMS" in order_depth:
        #     pos = positions.get("JAMS", 0)
        #     orders = self.general_order_machine(
        #         order_depth,
        #         "JAMS",
        #         pos,
        #     )
        #     result["JAMS"] = orders

        # if "DJEMBES" in order_depth:
        #     pos = positions.get("DJEMBES", 0)
        #     orders = self.general_order_machine(
        #         order_depth,
        #         "DJEMBES",
        #         pos,
        #     )
        #     result["DJEMBES"] = orders

        if "PICNIC_BASKET1" in order_depth:
            pos = positions.get("PICNIC_BASKET1", 0)
            orders = self.PICNIC_BASKET1_order(
                order_depth,
                pos
            )
            result["PICNIC_BASKET1"] = orders

        if "PICNIC_BASKET2" in order_depth:
            pos = positions.get("PICNIC_BASKET2", 0)
            orders = self.PICNIC_BASKET2_order(
                order_depth,
                pos
            )
            result["PICNIC_BASKET2"] = orders

        # if "DJEMBES" in order_depth\
        #         and "PICNIC_BASKET1" in order_depth\
        #             and "PICNIC_BASKET2" in order_depth:
        #     orders = self.SYNTH_DJEMBES_order(
        #         order_depth,
        #         positions
        #     )
        #     for order in orders:
        #         result.setdefault(order.symbol, []).append(order)

        if "VOLCANIC_ROCK" in order_depth:
            orders = self.VOLCANIC_ROCK_order(
                order_depth,
                positions,
                timestamp
            )
            for order in orders:
                result.setdefault(order.symbol, []).append(order)

        if "MAGNIFICENT_MACARONS" in order_depth:
            pos = positions.get("MAGNIFICENT_MACARONS", 0)
            pass

        # === PICKLE TRADER DATA ===
        traderData["kelp_historical"] = kelp_historical
        traderData["squid_ink_historical"] = squid_ink_historical
        traderData = jsonpickle.encode(traderData)

        return result, conversions, traderData
