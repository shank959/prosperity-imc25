# PROSPERITY

from typing import List
import string
import numpy as np
from datamodel import TradingState, OrderDepth, Order


class Trader:

    def fair_value(self, order_depth: OrderDepth):
        # volume weighted average price
        buy_orders = order_depth.buy_orders
        sell_orders = order_depth.sell_orders

        # # example order book
        # buy_orders = {
        #     "10": 10,
        #     "9": 20,
        #     "8": 30
        # }
        # sell_orders = {
        #     "11": 10,
        #     "12": 20,
        #     "13": 30
        # }
        # multiply each column with eachother and sum the result

        sell_keys = np.array([float(k) for k in sell_orders.keys()])
        sell_vals = np.array(list(sell_orders.values()), dtype=float)

        buy_keys = np.array([float(k) for k in buy_orders.keys()])
        buy_vals = np.array(list(buy_orders.values()), dtype=float)

        # Compute VWAP for each side
        vwap_sell = np.sum(sell_keys * sell_vals) / np.sum(sell_vals)
        vwap_buy  = np.sum(buy_keys  * buy_vals)  / np.sum(buy_vals)

        # fair value as average fo the two
        fair_value = (vwap_sell + vwap_buy) / 2

        return fair_value

    def run(self, state: TradingState):
        # Only method required. It takes all buy and sell orders for all symbols as an input, and outputs a list of orders to be sent
        print("traderData: " + state.traderData)
        print("Observations: " + str(state.observations))
        result = {}
        
        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []

            acceptable_price = self.fair_value(order_depth)  

            print("Acceptable price : " + str(acceptable_price))
            print("Buy Order depth : " + str(len(order_depth.buy_orders)) + ", Sell order depth : " + str(len(order_depth.sell_orders)))

            if len(order_depth.sell_orders) != 0:
                for ask, ask_amount in order_depth.sell_orders.items():
                    if int(ask) < acceptable_price:
                        print("BUY", str(-ask_amount) + "x", ask)
                        orders.append(Order(product, ask, -ask_amount))
                    else:
                        break

            if len(order_depth.buy_orders) != 0:
                # trade on buy orders higher than fair value
                best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]
                if int(best_bid) > acceptable_price:
                    print("SELL", str(best_bid_amount) + "x", best_bid)
                    orders.append(Order(product, best_bid, -best_bid_amount))
                else: break

            result[product] = orders


        traderData = "SAMPLE" # String value holding Trader state data required. It will be delivered as TradingState.traderData on next execution.

        conversions = 1
        return result, conversions, traderData
