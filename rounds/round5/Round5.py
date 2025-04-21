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
            },
            "JAMS": {
                "pos_lim": 350,
            },
            "DJEMBES": {
                "pos_lim": 60,
            },
            "PICNIC_BASKET1": {
                "pos_lim": 60,
                "take_buy_width": 1,
                "take_sell_width": 1,
                "make_buy_width": 2,
                "make_sell_width": 2,
            },
            "PICNIC_BASKET2": {
                "pos_lim": 100,
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
                "upper_threshold": 0.03894,
                "lower_threshold": -0.00803,
                "critical_boundary": 0.002,
            },
            "VOLCANIC_ROCK_VOUCHER_10000": {
                "pos_lim": 200,
                "take_buy_width": 1,
                "take_sell_width": 1,
                "make_buy_width": 2,
                "make_sell_width": 2,
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
                "storage_cost": 0.1,
                "take_buy_width": 1,
                "take_sell_width": 1,
                "make_width": 2,
                "critical_sunlight_index": 46,
                "conversion_limit": 10,
                "mm_size": 21,
                "size_cap": 30,
                "expected_holding_period": 10, #! need to determine through analysis
                "clear_width": 1,
            }
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

        return (mm_bid + mm_ask) / 2 if mm_bid and mm_ask else self.mid_price(product, order_depth)
    


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
        historical.append(fair_value)

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

        historical.append(fair_value)

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
    # PICNIC_BASKET SECTION
    # ============================

    def PICNIC_BASKET_order(
            self,
            order_depth: OrderDepth,
            positions: Dict[str, int]
    ) -> List[Order]:
        """
        """
        orders = []

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

        for product, data in info.items():
            pos = positions.get(product, 0)
            take_buy_width = self.PRODUCT_HYPERPARAMS[product].get(
                "take_buy_width", 1)
            take_sell_width = self.PRODUCT_HYPERPARAMS[product].get(
                "take_sell_width", 1)
            make_buy_width = self.PRODUCT_HYPERPARAMS[product].get(
                "make_buy_width", 1)
            make_sell_width = self.PRODUCT_HYPERPARAMS[product].get(
                "make_sell_width", 1)
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
                    if price >= fair_ask
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
                    if price <= fair_bid
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
            make_sell_amt = max(pos_lim + (pos - sell_amt), 20)
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
            make_buy_amt = max(pos_lim - (pos + buy_amt), 20)
            if make_buy_amt > 0:
                orders.append(Order(
                    product, bbbf + 1, make_buy_amt))
                print(f"[MAKE BID] {make_buy_amt} @ {bbbf + 1}")

        return orders
    
    # ============================
    # MAGNIFICENT_MACARONS SECTION
    # ============================

    def MAGNIFICENT_MACARONS_order(
            self,
            order_depth: OrderDepth,
            observation: Observation,
            pos: int,
    ) -> List[Order]:
        """
        """
        orders = []

        macaron_order_depth = order_depth["MAGNIFICENT_MACARONS"]
        pos_lim = self.PRODUCT_HYPERPARAMS["MAGNIFICENT_MACARONS"]["pos_lim"]
        take_buy_width = self.PRODUCT_HYPERPARAMS["MAGNIFICENT_MACARONS"].get(
            "take_buy_width", 1)
        take_sell_width = self.PRODUCT_HYPERPARAMS["MAGNIFICENT_MACARONS"].get(
            "take_sell_width", 1)
        make_width = self.PRODUCT_HYPERPARAMS["MAGNIFICENT_MACARONS"].get(
            "make_width", 2)
        conversion_limit = self.PRODUCT_HYPERPARAMS["MAGNIFICENT_MACARONS"].get(
            "conversion_limit", 10)
        position_threshold = self.PRODUCT_HYPERPARAMS["MAGNIFICENT_MACARONS"].get(
            "position_threshold", 20)
        conversion_amt = 0
        
        # DETERMINE CURRENT MODE
        sunlight_index = observation.sunlightIndex
        if sunlight_index > self.PRODUCT_HYPERPARAMS["MAGNIFICENT_MACARONS"]["critical_sunlight_index"]:
            mode = "STANDARD"
        else:
            mode = "PANIC"

        # === STANDARD MODE ===
        if mode == "STANDARD":
            # === EXCHANGE ARBITRAGE ORDERS === #! should be performed regardless of the mode?
            sell_amt = 0
            buy_amt = 0

            # implied_bid = observation.bidPrice - observation.transportFees - observation.exportTariff - self.PRODUCT_HYPERPARAMS["MAGNIFICENT_MACARONS"]["storage_cost"]
            # implied_ask = observation.askPrice + observation.transportFees + observation.importTariff
            # local_best_ask = min(order_depth["MAGNIFICENT_MACARONS"].sell_orders.keys(), default=float("inf"))
            # local_best_bid = max(order_depth["MAGNIFICENT_MACARONS"].buy_orders.keys(), default=0)

            # # HITTING THE SELL ORDERS (we buy local and export at implied bid)
            # if local_best_ask < implied_bid - take_buy_width:
            #     buy_qty = min(order_depth["MAGNIFICENT_MACARONS"].sell_orders[local_best_ask], conversion_limit)
            #     orders.append(Order("MAGNIFICENT_MACARONS", local_best_ask, buy_qty))
            #     conversion_amt -= buy_qty
            
            # # HITTING THE BUY ORDERS (we sell local and import at implied ask)
            # if local_best_bid > implied_ask + take_sell_width:
            #     sell_qty = min(order_depth["MAGNIFICENT_MACARONS"].buy_orders[local_best_bid], conversion_limit)
            #     orders.append(Order("MAGNIFICENT_MACARONS", local_best_bid, -sell_qty))
            #     conversion_amt += sell_qty

            # === CALCULATE FAIR VALUE ===
            fair_value = self.market_maker_mid("MAGNIFICENT_MACARONS", order_depth)
            # - self.PRODUCT_HYPERPARAMS["MAGNIFICENT_MACARONS"]["storage_cost"] * self.PRODUCT_HYPERPARAMS["MAGNIFICENT_MACARONS"]["expected_holding_period"] #! determine optimal adjustment to account for storage cost

            # === TAKE ===
            sell_amt = 0
            buy_amt = 0

            if macaron_order_depth.buy_orders:
                best_bid = max(macaron_order_depth.buy_orders.keys())
                if best_bid > fair_value + take_sell_width:
                    take_sell_amt = min(
                        macaron_order_depth.buy_orders[best_bid],
                        pos + pos_lim
                    )
                    sell_amt += take_sell_amt
                    orders.append(Order(
                        "MAGNIFICENT_MACARONS", round(best_bid), -take_sell_amt))
                    print(f"[TAKE OFFER] {take_sell_amt} @ {best_bid}")

            if macaron_order_depth.sell_orders:
                best_ask = min(macaron_order_depth.sell_orders.keys())
                if best_ask < fair_value - take_buy_width:
                    take_buy_amt = min(
                        abs(macaron_order_depth.sell_orders[best_ask]),
                        pos_lim - pos
                    )
                    buy_amt += take_buy_amt
                    orders.append(Order(
                        "MAGNIFICENT_MACARONS", round(best_ask), take_buy_amt))
                    print(f"[TAKE BID] {take_buy_amt} @ {best_ask}")

            pos_after_take = pos + buy_amt - sell_amt
            print(f"Position after take: {pos_after_take}")

            # === CLEAR ===
            clear_buy_amt = 0
            clear_sell_amt = 0
            clear_width = self.PRODUCT_HYPERPARAMS["MAGNIFICENT_MACARONS"].get(
                "clear_width", 1)
            fair_ask = math.ceil(fair_value)
            fair_bid = math.floor(fair_value)

            if pos_after_take > 0:
                # want to sell
                better_asks = [
                    price for price in order_depth["MAGNIFICENT_MACARONS"].buy_orders
                    if price >= fair_ask - clear_width
                ]
                if better_asks:
                    best_ask = max(better_asks)
                    clear_sell_amt = min(
                        order_depth["MAGNIFICENT_MACARONS"].buy_orders[best_ask],
                        pos_after_take,
                        pos_lim + (pos - sell_amt)
                    )
                    if clear_sell_amt > 0:
                        sell_amt += clear_sell_amt
                        orders.append(Order(
                            "MAGNIFICENT_MACARONS", best_ask, -clear_sell_amt))
                        print(f"[CLEAR OFFER]\
                              {clear_sell_amt} MAGNIFICENT_MACARONS @ {best_ask}")

            elif pos_after_take < 0:
                # want to buy
                better_bids = [
                    price for price in order_depth["MAGNIFICENT_MACARONS"].sell_orders
                    if price <= fair_bid + clear_width
                ]
                if better_bids:
                    best_bid = min(better_bids)
                    clear_buy_amt = min(
                        abs(order_depth["MAGNIFICENT_MACARONS"].sell_orders[best_bid]),
                        -pos_after_take,
                        pos_lim - (pos + buy_amt)
                    )
                    if clear_buy_amt > 0:
                        buy_amt += clear_buy_amt
                        orders.append(
                            Order("MAGNIFICENT_MACARONS", best_bid, clear_buy_amt))
                        print(f"[CLEAR BID]\
                              {clear_buy_amt} MAGNIFICENT_MACARONS @ {best_bid}")

            # === MAKE ===
            asks_above_fair_value = [
                price for price in macaron_order_depth.sell_orders
                if price > fair_value + make_width
            ]
            baaf = min(asks_above_fair_value)\
                if asks_above_fair_value\
                else math.ceil(fair_value + make_width)

            bids_below_fair_value = [
                price for price in macaron_order_depth.buy_orders
                if price < fair_value - make_width
            ]
            bbbf = max(bids_below_fair_value)\
                if bids_below_fair_value\
                else math.floor(fair_value - make_width)

            # if we ever cross our own quotes, back off to a non‐crossing spread
            best_bid = max(macaron_order_depth.buy_orders.keys(), default=0)
            best_ask = min(macaron_order_depth.sell_orders.keys(), default=float("inf"))
            if bbbf + make_width >= baaf - make_width:
                # either skip quoting or force a minimum spread of 1 tick
                baaf = math.ceil(best_ask - make_width)
                bbbf = math.floor(best_bid + make_width)


            size_cap = 30 #! play around with this
            make_sell_amt = pos_lim + (pos - sell_amt)
            make_buy_amt = pos_lim - (pos + buy_amt)
            size = min(make_sell_amt, make_buy_amt, size_cap)
            if make_sell_amt > 0:
                orders.append(Order(
                    "MAGNIFICENT_MACARONS", baaf - make_width, -size))

            if make_buy_amt > 0:
                orders.append(Order(
                    "MAGNIFICENT_MACARONS", bbbf + make_width, size))

        return orders


        # # === CLEAR POSITION ===
        # best_bid = max(order_depth["MAGNIFICENT_MACARONS"].buy_orders.keys(), default=0)
        # best_ask = min(order_depth["MAGNIFICENT_MACARONS"].sell_orders.keys(), default=float("inf"))
        # if pos >  position_threshold:
        #     # too long → sell immediately at best_bid
        #     qty = abs(min(pos - position_threshold, order_depth["MAGNIFICENT_MACARONS"].buy_orders[best_bid]))
        #     orders.append(Order("MAGNIFICENT_MACARONS", best_bid, -qty))
        # elif pos < -position_threshold:
        #     # too short → buy immediately at best_ask
        #     qty = abs(min(position_threshold + pos, order_depth["MAGNIFICENT_MACARONS"].sell_orders[best_ask]))
        #     orders.append(Order("MAGNIFICENT_MACARONS", best_ask,  qty))

            
        #     # === MAKE ORDERS ===
        #     # Determine fair value
        #     fair_value = self.market_maker_mid("MAGNIFICENT_MACARONS", order_depth)
        #     bids_above_fair_value = [
        #         price for price in order_depth["MAGNIFICENT_MACARONS"].buy_orders
        #         if price > fair_value + make_width
        #     ]
        #     if bids_above_fair_value:
        #         bbaf = max(bids_above_fair_value)
        #     else:
        #         bbaf = math.floor(fair_value + make_width)
        #     asks_below_fair_value = [
        #         price for price in order_depth["MAGNIFICENT_MACARONS"].sell_orders
        #         if price < fair_value - make_width
        #     ]
        #     best_ask = min(order_depth["MAGNIFICENT_MACARONS"].sell_orders, default=float("inf"))

        #     # simple one‐tick spread from NBBO
        #     make_width = self.PRODUCT_HYPERPARAMS["MAGNIFICENT_MACARONS"].get("make_width", 1)  # e.g. 1 tick
        #     bid_price = best_bid + make_width
        #     ask_price = best_ask - make_width

        #     # if we ever cross our own quotes, back off to a non‐crossing spread
        #     if bid_price >= ask_price:
        #         # either skip quoting or force a minimum spread of 1 tick
        #         bid_price = best_bid
        #         ask_price = best_ask

        #     # size as before
        #     size = min(pos_lim - pos, int(min(
        #         order_depth["MAGNIFICENT_MACARONS"].buy_orders[best_bid],
        #         abs(order_depth["MAGNIFICENT_MACARONS"].sell_orders[best_ask]))))

        #     # post passive quotes
        #     orders.append(Order("MAGNIFICENT_MACARONS", bid_price,  size))
        #     orders.append(Order("MAGNIFICENT_MACARONS", ask_price, -size))




        # return orders, conversion_amt


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
        # if "RAINFOREST_RESIN" in order_depth:
        #     pos = positions.get("RAINFOREST_RESIN", 0)
        #     result["RAINFOREST_RESIN"] = self.RAINFOREST_RESIN_order(
        #         order_depth,
        #         pos
        #     )

        # if "KELP" in order_depth:
        #     pos = positions.get("KELP", 0)
        #     result["KELP"], kelp_historical = self.KELP_order(
        #         order_depth,
        #         pos,
        #         kelp_historical
        #     )

        # if "SQUID_INK" in order_depth:
        #     pos = positions.get("SQUID_INK", 0)
        #     result["SQUID_INK"], squid_ink_historical = self.SQUID_INK_order(
        #         order_depth,
        #         pos,
        #         squid_ink_historical
        #     )

        if "PICNIC_BASKET1" in order_depth:
            pos = positions.get("PICNIC_BASKET1", 0)
            pass

        if "PICNIC_BASKET2" in order_depth:
            pos = positions.get("PICNIC_BASKET2", 0)
            pass

        # if "VOLCANIC_ROCK" in order_depth:
        #     orders = self.VOLCANIC_ROCK_order(
        #         order_depth,
        #         positions,
        #         timestamp
        #     )
        #     for order in orders:
        #         result.setdefault(order.symbol, []).append(order)

        if "MAGNIFICENT_MACARONS" in order_depth:
            pos = positions.get("MAGNIFICENT_MACARONS", 0)
            observation = state.observations.conversionObservations["MAGNIFICENT_MACARONS"]
            orders = self.MAGNIFICENT_MACARONS_order(
                order_depth,
                observation,
                pos
            )
            # print(f"sunlight index: {state.observations.conversionObservations['MAGNIFICENT_MACARONS'].sunlightIndex}")
            result["MAGNIFICENT_MACARONS"] = orders
            # print(f"orders: {result['MAGNIFICENT_MACARONS']}")


        # === PICKLE TRADER DATA ===
        traderData["kelp_historical"] = kelp_historical
        traderData["squid_ink_historical"] = squid_ink_historical
        traderData = jsonpickle.encode(traderData)

        return result, conversions, traderData
