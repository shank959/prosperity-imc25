from datamodel import OrderDepth, UserId, TradingState, Order, Trade
from typing import Dict
import jsonpickle
import numpy as np
import math
from math import log, sqrt
from statistics import NormalDist


class Trader:
    def midprice(self, product, order_depth: OrderDepth) -> float:
        best_bid = max(order_depth[product].buy_orders.keys())\
            if order_depth[product].buy_orders else 0
        best_ask = min(order_depth[product].sell_orders.keys())\
            if order_depth[product].sell_orders else 0
        return (best_bid + best_ask) / 2 if best_bid and best_ask\
            else best_bid + best_ask

    def clear_position(self, product: str,
                       positions: Dict[str, int],
                       average_entry_price: Dict[str, Dict[str, float]],
                       order_depth: OrderDepth):
        """
        Clear position of a product
        """
        if product in positions:
            if positions[product] < 0:
                # want to buy at lower than average entry price
                best_ask = min(order_depth[product].sell_orders.keys())\
                    if order_depth[product].sell_orders else 0
                if best_ask < average_entry_price[product] - 10:
                    print("buying at", best_ask, "to clear position")
                    return Order(product, best_ask, positions[product])
            elif positions[product] > 0:
                # want to sell at higher than average entry price
                best_bid = max(order_depth[product].buy_orders.keys())\
                    if order_depth[product].buy_orders else 0
                if best_bid > average_entry_price[product] + 10:
                    print("selling at", best_bid, "to clear position")
                    return Order(product, best_bid, positions[product])
        return None

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

        info["VOLCANIC_ROCK_VOUCHER_9500"]["upper_threshold"] = 0.003
        info["VOLCANIC_ROCK_VOUCHER_9500"]["lower_threshold"] = -0.015

        info["VOLCANIC_ROCK_VOUCHER_9750"]["upper_threshold"] = 0.03
        info["VOLCANIC_ROCK_VOUCHER_9750"]["lower_threshold"] = -0.01

        info["VOLCANIC_ROCK_VOUCHER_10000"]["upper_threshold"] = 0.003
        info["VOLCANIC_ROCK_VOUCHER_10000"]["lower_threshold"] = -0.01

        info["VOLCANIC_ROCK_VOUCHER_10250"]["upper_threshold"] = 0.003
        info["VOLCANIC_ROCK_VOUCHER_10250"]["lower_threshold"] = -0.015

        info["VOLCANIC_ROCK_VOUCHER_10500"]["upper_threshold"] = 0.01
        info["VOLCANIC_ROCK_VOUCHER_10500"]["lower_threshold"] = 0

        for product in info:
            if info[product]["mispricing"] > info[product]["upper_threshold"]:
                # get the prices
                best_voucher_bid = max(order_depth[product].buy_orders.keys())\
                    if order_depth[product].buy_orders else 0
                best_underlying_ask = min(order_depth["VOLCANIC_ROCK"].sell_orders.keys())\
                    if order_depth["VOLCANIC_ROCK"].sell_orders else 0

                # how much we can sell is determined by our
                # 1) position limit
                # 2) how much underlying we can buy
                # 2a) position limit of underlying
                # 2b) available liquidity of underlying
                # 3) liquidity of the best bid
                # 4) aggression

                sell_amount = min(
                    positions.get(product, 0) + 200, # limit of vouchers
                    abs(order_depth[product].buy_orders.get(best_voucher_bid, 0)),
                    (400 - positions.get("VOLCANIC_ROCK", 0) - underlying_order_count) // info[product]["delta"],
                    abs(order_depth["VOLCANIC_ROCK"].sell_orders.get(best_underlying_ask, 0)),
                    aggression
                )

                if sell_amount > 0:
                    orders.append(Order(
                        product,
                        best_voucher_bid,
                        - round(sell_amount)
                    ))
                    orders.append(Order(
                        "VOLCANIC_ROCK",
                        best_underlying_ask,
                        round(sell_amount * info[product]["delta"])
                    ))
                    underlying_order_count += round(sell_amount * info[product]["delta"])

            elif info[product]["mispricing"] < info[product]["lower_threshold"]:
                # get the prices
                best_voucher_ask = min(order_depth[product].sell_orders.keys())\
                    if order_depth[product].sell_orders else 0
                best_underlying_bid = max(order_depth["VOLCANIC_ROCK"].buy_orders.keys())\
                    if order_depth["VOLCANIC_ROCK"].buy_orders else 0

                # how much we can buy is determined by our
                # 1) position limit
                # 2) how much underlying we can sell
                # 2a) position limit of underlying
                # 2b) available liquidity of underlying
                # 3) liquidity of the best ask
                # 4) aggression

                buy_amount = min(
                    - positions.get(product, 0) + 200, # limit of vouchers
                    abs(order_depth[product].sell_orders[best_voucher_ask]),
                    (400 + positions.get("VOLCANIC_ROCK", 0) + underlying_order_count) // info[product]["delta"],
                    abs(order_depth["VOLCANIC_ROCK"].buy_orders.get(best_underlying_bid, 0)),
                    aggression
                )

                if buy_amount > 0:
                    orders.append(Order(
                        product,
                        best_voucher_ask,
                        round(buy_amount)
                    ))
                    orders.append(Order(
                        "VOLCANIC_ROCK",
                        best_underlying_bid,
                        - round(buy_amount * info[product]["delta"])
                    ))
                    underlying_order_count -= round(buy_amount * info[product]["delta"])

        # clear position
        for product in info:
            order = self.clear_position(
                product,
                positions,
                average_entry_price,
                order_depth
            )
            if order:
                orders.append(order)

        return orders

    def run(self, state: TradingState):
        # need to track pnl of each product, so could be position plus average entry price
        positions = state.position
        order_depth = state.order_depths
        timestamp = state.timestamp
        own_trades = state.own_trades
        orders = []
        result = {}
        conversions = 0

        # if state.traderData:
        #     trader_data = jsonpickle.decode(state.traderData)
        #     average_entry_prices = trader_data["average_entry_prices"]
        #     for product, price in average_entry_prices.items():
        #         weighted_entry_price = 0
        #         trade_size = 0
        #         for trade in own_trades.get(product, []):
        #             weighted_entry_price += trade.price * trade.quantity
        #             trade_size += trade.quantity
        #         if trade_size:
        #             new_total_quantity = trade_size + positions[product]
        #             new_total_cost = weighted_entry_price\
        #                 + average_entry_prices[product] * positions[product]
        #             if new_total_quantity != 0:
        #                 average_entry_prices[product] = new_total_cost / new_total_quantity
        #             else:
        #                 average_entry_prices[product] = 0
        #     # print("Average entry prices: ", average_entry_prices)

        if state.traderData:
            trader_data = jsonpickle.decode(state.traderData)
            average_entry_prices = trader_data["average_entry_prices"]
            for product, price in average_entry_prices.items():
                prev_position = positions.get(product, 0) - sum(trade.quantity for trade in own_trades.get(product, []))
                net_position = positions.get(product, 0)

                # If direction flipped (or position went from 0), reset
                if prev_position * net_position <= 0:
                    # use only the new trades to compute entry price
                    total_qty = 0
                    total_cost = 0
                    for trade in own_trades.get(product, []):
                        if trade.quantity * net_position > 0:  # same direction
                            total_cost += trade.price * abs(trade.quantity)
                            total_qty += abs(trade.quantity)
                    average_entry_prices[product] = total_cost / total_qty if total_qty else 0
                else:
                    # continue normal weighted average
                    total_cost = average_entry_prices[product] * abs(prev_position)
                    for trade in own_trades.get(product, []):
                        total_cost += trade.price * abs(trade.quantity)
                    average_entry_prices[product] = total_cost / abs(net_position) if net_position else 0         

        else:
            known_products = list(state.order_depths.keys())
            average_entry_prices = {product: 0 for product in known_products}

        # Generate orders
        if "VOLCANIC_ROCK" in order_depth:
            orders += self.volc_orders(order_depth, positions, timestamp,
                                       average_entry_prices, aggression=5)

        for order in orders:
            result[order.symbol] = result.get(order.symbol, []) + [order]

        next_trader_data = {
            "average_entry_prices": average_entry_prices
        }
        traderData = jsonpickle.encode(next_trader_data)

        return result, conversions, traderData
