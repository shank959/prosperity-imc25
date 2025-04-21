from datamodel import OrderDepth, UserId, TradingState, Order, Trade
from typing import Dict, List
import jsonpickle
import numpy as np
import math
from math import log, sqrt, exp
from statistics import NormalDist
import random


class Trader:
    def midprice(self, product, order_depth: OrderDepth) -> float:
        best_bid = max(order_depth[product].buy_orders.keys())\
            if order_depth[product].buy_orders else 0
        best_ask = min(order_depth[product].sell_orders.keys())\
            if order_depth[product].sell_orders else 0
        return (best_bid + best_ask) / 2 if best_bid and best_ask\
            else best_bid + best_ask

    def market_make(
        self,
        fair: float,
        product: str,
        positions: Dict[str, int],
        pos_lim: int,
        order_depth: OrderDepth
    ) -> List[Order]:
        """
        Place bid/ask quotes skewed by inventory.
        """
        orders: List[Order] = []
        pos = positions.get(product, 0)

        best_bid = max(order_depth[product].buy_orders.keys())\
            if order_depth[product].buy_orders else 0
        best_ask = min(order_depth[product].sell_orders.keys())\
            if order_depth[product].sell_orders else 0

        width = best_ask - best_bid if best_bid and best_ask\
            else 0

        half_spread = max(round(width / 2), 1)
        inv_ratio = pos / pos_lim  # in [-1, 1]
        skew = inv_ratio * half_spread

        # Skew quotes to manage inventory
        bid_price = round(fair - half_spread + skew)
        ask_price = round(fair + half_spread - skew)

        # Determine quote sizes
        max_buy = pos_lim - pos
        max_sell = pos_lim + pos
        base_size = 10
        bid_qty = min(base_size, max_buy) if max_buy > 0 else 0
        ask_qty = min(base_size, max_sell) if max_sell > 0 else 0

        # Append bid and ask orders
        if bid_qty > 0:
            orders.append(Order(product, bid_price, bid_qty))
        if ask_qty > 0:
            orders.append(Order(product, ask_price, -ask_qty))

        return orders

    def clear_position(self, product: str,
                       positions: Dict[str, int],
                       pos_lim: int,
                       order_depth: OrderDepth) -> List[Order]:
        orders = []
        position = positions.get(product, 0)
        best_ask = min(order_depth[product].sell_orders.keys())\
            if order_depth[product].sell_orders else 0
        best_bid = max(order_depth[product].buy_orders.keys())\
            if order_depth[product].buy_orders else 0
        mid = (best_ask + best_bid) / 2 if best_ask and best_bid\
            else best_ask + best_bid
        spread = best_ask - best_bid

        if position > pos_lim - 0.2 * pos_lim:
            # spread_factor = round(4 * (pos_lim - position) / pos_lim)
            # orders.append(Order(product, round(best_bid + spread_factor * spread/2), min(4, position)))
            # orders.append(Order(product, round(best_ask), - min(4, position)))
            orders.append(Order(product, round(mid), - min(10, position)))
        elif position < -pos_lim + 0.2 * pos_lim:
            # spread_factor = round(4 * (pos_lim + position) / pos_lim)
            # orders.append(Order(product, round(best_bid - spread_factor * spread/2), - max(-4, position)))
            # orders.append(Order(product, round(best_ask), max(-4, position)))
            orders.append(Order(product, round(mid), max(-10, position)))
        return orders

    # IV calculation Newton-Raphson method or bisection method
    def fallback_bisection_iv(self, St, Vt, K, TTE, low=1e-4, high=1, tol=1e-4):
        N = NormalDist()
        for _ in range(10):
            mid = (low + high) / 2
            d1 = (math.log(St/K) + 0.5 * mid**2 * TTE) / (mid * math.sqrt(TTE))
            d2 = d1 - mid * math.sqrt(TTE)
            delta = N.cdf(d1)
            price = St * delta - K * N.cdf(d2)
            if abs(price - Vt) < tol:
                return mid, delta
            if price > Vt:
                high = mid
            else:
                low = mid
        return mid, delta 

    def black_scholes_implied_vol(self, St, Vt, K, TTE, tol=1e-8, max_iter=50):
        N = NormalDist()
        sigma = 0.2
        for _ in range(max_iter):
            d1 = (math.log(St / K) + 0.5 * sigma**2 * TTE)\
                / (sigma * math.sqrt(TTE))
            d2 = d1 - sigma * math.sqrt(TTE)
            delta = N.cdf(d1)
            price = St * delta - K * N.cdf(d2)
            vega = St * N.pdf(d1) * math.sqrt(TTE)
            diff = price - Vt
            if vega < 1e-8:
                return self.fallback_bisection_iv(St, Vt, K, TTE)
            if abs(diff) < tol:
                return sigma, delta
            sigma -= diff / vega
        return sigma, delta

    def volcanic_rock_iv_fit(self, order_depth: OrderDepth, timestamp: int):
        """
        Fit smile and generate signal
        """
        strikes = [9500, 9750, 10000, 10250, 10500]
        TTE = (4 * 1e6 - timestamp)/(1e6 * 365)
        St = self.midprice("VOLCANIC_ROCK", order_depth)

        # first calculate price, IV, and moneyness
        info = {}
        for strike in strikes:
            product = f"VOLCANIC_ROCK_VOUCHER_{strike}"
            price = self.midprice(product, order_depth)
            iv, delta = self.black_scholes_implied_vol(
                St,
                price,
                strike,
                TTE
            )
            info[product] = {
                "price": price,
                "iv": iv,
                "delta": delta,
                "moneyness": log(strike/St) / sqrt(TTE)
            }

        # then fit the IV curve
        coeffs = np.polyfit(
            [info[product]["moneyness"] for product in info],
            [info[product]["iv"] for product in info],
            2
        )

        # calculate the fitted IV for the volcanic rock
        for product in info:
            info[product]["iv_fit"] =\
                coeffs[0] * info[product]["moneyness"]**2\
                + coeffs[1] * info[product]["moneyness"]\
                + coeffs[2]

        # calculate the mispricing
        for product in info:
            info[product]["mispricing"] =\
                (info[product]["iv"] - info[product]["iv_fit"])

        # calculate the fair price
        for product, data in info.items():
            N = NormalDist()
            S = St
            K = int(product.split("_")[-1])
            sigma = data["iv_fit"]
            T = TTE
            r = 0

            if T <= 0 or sigma <= 0:
                data["fair_price"] = 0.0
                continue

            d1 = (log(S/K) + 0.5 * sigma**2 * T) / (sigma * sqrt(T))
            d2 = d1 - sigma * sqrt(T)
            call_price = S * N.cdf(d1) - K * exp(-r * T) * N.cdf(d2)
            data["fair_price"] = call_price

        return info

    def volc_orders(self, order_depth: OrderDepth,
                    positions: Dict[str, int], timestamp: int,
                    average_entry_price: Dict[str, Dict[str, float]],
                    aggression=float("inf")):
        """
        Generate orders for volcanic rock
        """
        orders = []
        info = self.volcanic_rock_iv_fit(order_depth, timestamp)
        underlying_order_count = 0
        underlying_bid_count = 0
        underlying_ask_count = 0

        underlying_position = positions.get("VOLCANIC_ROCK", 0)

        best_underlying_ask = min(order_depth["VOLCANIC_ROCK"].sell_orders.keys())\
            if order_depth["VOLCANIC_ROCK"].sell_orders else 0
        best_underlying_ask_amount = abs(order_depth["VOLCANIC_ROCK"].sell_orders.get(
            best_underlying_ask, 0))

        best_underlying_bid = max(order_depth["VOLCANIC_ROCK"].buy_orders.keys())\
            if order_depth["VOLCANIC_ROCK"].buy_orders else 0
        best_underlying_bid_amount = abs(order_depth["VOLCANIC_ROCK"].buy_orders.get(
            best_underlying_bid, 0))

        info["VOLCANIC_ROCK_VOUCHER_9500"]["upper_threshold"] = 0.00322
        info["VOLCANIC_ROCK_VOUCHER_9500"]["lower_threshold"] = -0.01665
        info["VOLCANIC_ROCK_VOUCHER_9500"]["critical_boundary"] = 0.001

        info["VOLCANIC_ROCK_VOUCHER_9750"]["upper_threshold"] = 0.03894
        info["VOLCANIC_ROCK_VOUCHER_9750"]["lower_threshold"] = -0.00803
        info["VOLCANIC_ROCK_VOUCHER_9750"]["critical_boundary"] = 0.001

        info["VOLCANIC_ROCK_VOUCHER_10000"]["upper_threshold"] = 0.00534
        info["VOLCANIC_ROCK_VOUCHER_10000"]["lower_threshold"] = -0.01640
        info["VOLCANIC_ROCK_VOUCHER_10000"]["critical_boundary"] = 0.001

        info["VOLCANIC_ROCK_VOUCHER_10250"]["upper_threshold"] = 0.00502
        info["VOLCANIC_ROCK_VOUCHER_10250"]["lower_threshold"] = -0.02194
        info["VOLCANIC_ROCK_VOUCHER_10250"]["critical_boundary"] = 0.001

        info["VOLCANIC_ROCK_VOUCHER_10500"]["upper_threshold"] = float("inf")
        info["VOLCANIC_ROCK_VOUCHER_10500"]["lower_threshold"] = float("-inf")
        info["VOLCANIC_ROCK_VOUCHER_10500"]["critical_boundary"] = 0.0

        keys = list(info.keys())
        random.shuffle(keys)
        for product in keys:
            voucher_position = positions.get(product, 0)
            delta = info[product]["delta"]

            # sell if IV of voucher is higher than fitted IV by upper threshold
            if info[product]["mispricing"] > info[product]["upper_threshold"]:
                # get the prices
                best_voucher_bid = max(order_depth[product].buy_orders.keys())\
                    if order_depth[product].buy_orders else 0
                best_voucher_bid_amount = abs(order_depth[product].buy_orders.get(best_voucher_bid, 0))

                # how much we can sell is determined by our
                # 1) position limit
                # 2) how much underlying we can buy
                # 2a) position limit of underlying
                # 2b) available liquidity of underlying
                # 3) liquidity of the best bid
                # 4) aggression

                sell_amount = min(
                    voucher_position + 200,
                    best_voucher_bid_amount,
                    # (400 - underlying_position - underlying_order_count) // delta,
                    # max((best_underlying_ask_amount - underlying_ask_count), 0) // delta,
                    aggression
                )

                if sell_amount > 0 and round(sell_amount * delta) > 0:
                    orders.append(Order(
                        product,
                        best_voucher_bid,
                        - round(sell_amount)
                    ))
                    # orders.append(Order(
                    #     "VOLCANIC_ROCK",
                    #     best_underlying_ask,
                    #     round(sell_amount * delta)
                    # ))

                    underlying_order_count += round(sell_amount * delta)
                    underlying_ask_count += round(sell_amount * delta)

            # buy if IV of voucher is lower than fitted IV by lower threshold
            elif info[product]["mispricing"] < info[product]["lower_threshold"]:
                # get the prices
                best_voucher_ask = min(order_depth[product].sell_orders.keys())\
                    if order_depth[product].sell_orders else 0
                best_voucher_ask_amount = abs(order_depth[product].sell_orders.get(
                    best_voucher_ask, 0))

                # how much we can buy is determined by our
                # 1) position limit
                # 2) how much underlying we can sell
                # 2a) position limit of underlying
                # 2b) available liquidity of underlying
                # 3) liquidity of the best ask
                # 4) aggression

                buy_amount = min(
                    200 - voucher_position,
                    abs(best_voucher_ask_amount),
                    # (400 + underlying_position + underlying_order_count) // delta,
                    # max(best_underlying_bid_amount - underlying_bid_count, 0) // delta,
                    aggression
                )

                if round(buy_amount) > 0 and round(buy_amount * delta) > 0:
                    orders.append(Order(
                        product,
                        best_voucher_ask,
                        round(buy_amount)
                    ))
                    # orders.append(Order(
                    #     "VOLCANIC_ROCK",
                    #     best_underlying_bid,
                    #     - round(buy_amount * delta)
                    # ))
                    underlying_order_count -= round(buy_amount * delta)
                    underlying_bid_count += round(buy_amount * delta)

            elif abs(info[product]["mispricing"]) < info[product]["critical_boundary"]:
                # print(f"Critical boundary crossed for {product} at {timestamp}")
                # offload
                if voucher_position > 0:
                    best_voucher_bid = max(order_depth[product].buy_orders.keys())\
                        if order_depth[product].buy_orders else 0
                    best_voucher_bid_amount = abs(order_depth[product].buy_orders.get(
                        best_voucher_bid, 0))

                    sell_amount = min(
                        voucher_position,
                        best_voucher_bid_amount,
                        # (400 - underlying_position - underlying_order_count) // delta,
                        # max((best_underlying_ask_amount - underlying_ask_count), 0) // delta,
                        aggression
                    )

                    if sell_amount > 0 and round(sell_amount * delta) > 0:
                        # print(f"Offloading {sell_amount} of {product} at {timestamp}")
                        # print(f"Current position: {voucher_position - sell_amount}")
                        orders.append(Order(
                            product,
                            best_voucher_bid,
                            - round(sell_amount)
                        ))
                        # orders.append(Order(
                        #     "VOLCANIC_ROCK",
                        #     best_underlying_ask,
                        #     round(sell_amount * delta)
                        # ))

                        underlying_order_count += round(sell_amount * delta)
                        underlying_ask_count += round(sell_amount * delta)

                elif voucher_position < 0:
                    best_voucher_ask = min(order_depth[product].sell_orders.keys())\
                        if order_depth[product].sell_orders else 0
                    best_voucher_ask_amount = abs(order_depth[product].sell_orders.get(
                        best_voucher_ask, 0))

                    buy_amount = min(
                        - voucher_position,
                        abs(best_voucher_ask_amount),
                        # (400 + underlying_position + underlying_order_count) // delta,
                        # max(best_underlying_bid_amount - underlying_bid_count, 0) // delta,
                        aggression
                    )

                    if round(buy_amount) > 0 and round(buy_amount * delta) > 0:
                        # print(f"Offloading {buy_amount} of {product} at {timestamp}")
                        # print(f"Current position: {voucher_position + buy_amount}")
                        orders.append(Order(
                            product,
                            best_voucher_ask,
                            round(buy_amount)
                        ))
                        # orders.append(Order(
                        #     "VOLCANIC_ROCK",
                        #     best_underlying_bid,
                        #     - round(buy_amount * delta)
                        # ))
                        underlying_order_count -= round(buy_amount * delta)
                        underlying_bid_count += round(buy_amount * delta)

        # # market make
        # for product in info:
        #     if product in order_depth:
        #         orders += self.market_make(
        #             info[product]["fair_price"],
        #             product,
        #             positions,
        #             200,
        #             order_depth
        #         )

        # # clear position
        # for product in info:
        #     order = self.clear_position(
        #         product,
        #         positions,
        #         200,
        #         # average_entry_price,
        #         order_depth
        #     )
        #     if order:
        #         orders.extend(order)

        # order = self.clear_position(
        #     "VOLCANIC_ROCK",
        #     positions,
        #     400,
        #     order_depth
        # )
        # if order:
        #     orders.extend(order)

        # aggregate orders and print current position and total quantity for each product
        # order_dict = {}
        # for order in orders:
        #     if order.symbol not in order_dict:
        #         order_dict[order.symbol] = 0
        #     order_dict[order.symbol] += order.quantity

        # for product in order_dict:
        #     print(f"positions: {positions.get(product, 0)}, order {product} quantity {order_dict[product]}")

        return orders

    def run(self, state: TradingState):
        positions = state.position
        order_depth = state.order_depths
        timestamp = state.timestamp
        own_trades = state.own_trades
        orders = []
        result = {}
        conversions = 0

        # Initialise or load trader state
        if state.traderData:
            trader_data = jsonpickle.decode(state.traderData)
            cost_basis = trader_data['cost_basis']
            prev_positions = trader_data['prev_position']
        else:
            cost_basis = {product: 0.0 for product in state.order_depths}
            prev_positions = {product: 0 for product in state.order_depths}

        average_entry_prices = {}

        # Update cost basis and average entry price
        for product in state.order_depths:
            prev_pos = prev_positions.get(product, 0)
            cost = cost_basis.get(product, 0.0)
            fills = own_trades.get(product, [])

            # compute net & gross fills
            net_qty = sum(t.quantity for t in fills)         # signed
            gross_qty = sum(abs(t.quantity) for t in fills)    # always ≥ 0

            new_pos = prev_pos + net_qty

            if gross_qty == 0:
                # no change
                average_entry_prices[product] = cost / prev_pos if prev_pos else 0.0
                continue

            # VWAP on absolute quantities
            avg_price = sum(t.price * abs(t.quantity) for t in fills) / gross_qty

            # same‑side add or trim (never cross zero)
            if prev_pos == 0 or prev_pos * new_pos > 0:
                cost += net_qty * avg_price

            else:
                # you've flipped through zero
                # the part that closed prev_pos is realised PnL
                # the remainder opens new_pos -> cost basis = new_pos * avg_price
                cost = new_pos * avg_price

            if new_pos == 0:
                cost = 0.0

            cost_basis[product] = cost
            average_entry_prices[product] = cost / new_pos if new_pos else 0.0

        # print("Positions:", positions)
        # print("Cost basis:", cost_basis)
        # print("Average entry prices:", average_entry_prices)

        # Generate orders
        if "VOLCANIC_ROCK" in order_depth:
            orders += self.volc_orders(order_depth, positions, timestamp,
                                        average_entry_prices, aggression=float("inf"))

            # TEST FOR 10500 CHEESE
            if "VOLCANIC_ROCK_VOUCHER_10500" in order_depth:
                if order_depth["VOLCANIC_ROCK_VOUCHER_10500"].buy_orders:
                    best_bid = max(order_depth["VOLCANIC_ROCK_VOUCHER_10500"].buy_orders.keys())
                    if best_bid > 3.5:
                        orders.append(Order("VOLCANIC_ROCK_VOUCHER_10500", best_bid, - 30))
                if order_depth["VOLCANIC_ROCK_VOUCHER_10500"].sell_orders:
                    best_ask = min(order_depth["VOLCANIC_ROCK_VOUCHER_10500"].sell_orders.keys())
                    if best_ask < 2.6:
                        orders.append(Order("VOLCANIC_ROCK_VOUCHER_10500", best_ask, 30))

        for order in orders:
            result.setdefault(order.symbol, []).append(order)

        trader_data = jsonpickle.encode({
            'cost_basis': cost_basis,
            'prev_position': positions
        })

        return result, conversions, trader_data
