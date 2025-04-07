from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import string
import jsonpickle
import numpy as np
import math

class Trader:
    def __init__(self):
        self.KELP_prices = []
        self.KELP_vwap = []

    def rfr_orders(self, order_depth: OrderDepth, fair_value: int, width: int, position: int, position_limit: int) -> List[Order]:
        orders: List[Order] = []

        buy_order_volume = 0
        sell_order_volume = 0
        # mm_ask = min([price for price in order_depth.sell_orders.keys() if abs(order_depth.sell_orders[price]) >= 20])
        # mm_bid = max([price for price in order_depth.buy_orders.keys() if abs(order_depth.buy_orders[price]) >= 20])

        baaf = min([price for price in order_depth.sell_orders.keys() if price > fair_value + 1])
        bbbf = max([price for price in order_depth.buy_orders.keys() if price < fair_value - 1])

        if len(order_depth.sell_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -1*order_depth.sell_orders[best_ask]
            if best_ask < fair_value:
                quantity = min(best_ask_amount, position_limit - position) # max amt to buy 
                if quantity > 0:
                    orders.append(Order("RAINFOREST_RESIN", best_ask, quantity)) 
                    buy_order_volume += quantity

        if len(order_depth.buy_orders) != 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]
            if best_bid > fair_value:
                quantity = min(best_bid_amount, position_limit + position) # should be the max we can sell 
                if quantity > 0:
                    orders.append(Order("RAINFOREST_RESIN", best_bid, -1 * quantity))
                    sell_order_volume += quantity

        buy_order_volume, sell_order_volume = self.clear_position_order(orders, order_depth, position, position_limit, "RAINFOREST_RESIN", buy_order_volume, sell_order_volume, fair_value, 1)

        buy_quantity = position_limit - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order("RAINFOREST_RESIN", bbbf + 1, buy_quantity))  # Buy order

        sell_quantity = position_limit + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order("RAINFOREST_RESIN", baaf - 1, -sell_quantity))  # Sell order

        return orders

    def clear_position_order(self, orders: List[Order], order_depth: OrderDepth, position: int, position_limit: int, product: str, buy_order_volume: int, sell_order_volume: int, fair_value: float, width: int) -> List[Order]:

        position_after_take = position + buy_order_volume - sell_order_volume
        fair = round(fair_value)
        fair_for_bid = math.floor(fair_value)
        fair_for_ask = math.ceil(fair_value)
        # fair_for_ask = fair_for_bid = fair

        buy_quantity = position_limit - (position + buy_order_volume)
        sell_quantity = position_limit + (position - sell_order_volume)

        if position_after_take > 0:
            if fair_for_ask in order_depth.buy_orders.keys():
                clear_quantity = min(order_depth.buy_orders[fair_for_ask], position_after_take)
                # clear_quantity = position_after_take
                sent_quantity = min(sell_quantity, clear_quantity)
                orders.append(Order(product, fair_for_ask, -abs(sent_quantity)))
                sell_order_volume += abs(sent_quantity)

        if position_after_take < 0:
            if fair_for_bid in order_depth.sell_orders.keys():
                clear_quantity = min(abs(order_depth.sell_orders[fair_for_bid]), abs(position_after_take))
                # clear_quantity = abs(position_after_take)
                sent_quantity = min(buy_quantity, clear_quantity)
                orders.append(Order(product, fair_for_bid, abs(sent_quantity)))
                buy_order_volume += abs(sent_quantity)

        return buy_order_volume, sell_order_volume

    # Method: mid_price,
    def KELP_fair_value(self, order_depth: OrderDepth, method = "mid_price", min_vol = 0) -> float:
        if method == "mid_price":
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            mid_price = (best_ask + best_bid) / 2
            return mid_price
        elif method == "mid_price_with_vol_filter":
            if len([price for price in order_depth.sell_orders.keys() if abs(order_depth.sell_orders[price]) >= min_vol]) ==0 or len([price for price in order_depth.buy_orders.keys() if abs(order_depth.buy_orders[price]) >= min_vol]) ==0:
                best_ask = min(order_depth.sell_orders.keys())
                best_bid = max(order_depth.buy_orders.keys())
                mid_price = (best_ask + best_bid) / 2
                return mid_price
            else:
                best_ask = min([price for price in order_depth.sell_orders.keys() if abs(order_depth.sell_orders[price]) >= min_vol])
                best_bid = max([price for price in order_depth.buy_orders.keys() if abs(order_depth.buy_orders[price]) >= min_vol])
                mid_price = (best_ask + best_bid) / 2
            return mid_price

    def KELP_orders(self, order_depth: OrderDepth, timespan:int, width: float, KELP_take_width: float, position: int, position_limit: int) -> List[Order]:
        orders: List[Order] = []

        buy_order_volume = 0
        sell_order_volume = 0

        if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:    

            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            filtered_ask = [price for price in order_depth.sell_orders.keys() if abs(order_depth.sell_orders[price]) >= 15]
            filtered_bid = [price for price in order_depth.buy_orders.keys() if abs(order_depth.buy_orders[price]) >= 15]
            mm_ask = min(filtered_ask) if len(filtered_ask) > 0 else best_ask
            mm_bid = max(filtered_bid) if len(filtered_bid) > 0 else best_bid
            
            mmmid_price = (mm_ask + mm_bid) / 2
            self.KELP_prices.append(mmmid_price)

            volume = -1 * order_depth.sell_orders[best_ask] + order_depth.buy_orders[best_bid]
            vwap = (best_bid * (-1) * order_depth.sell_orders[best_ask] + best_ask * order_depth.buy_orders[best_bid]) / volume
            self.KELP_vwap.append({"vol": volume, "vwap": vwap})

            if len(self.KELP_vwap) > timespan:
                self.KELP_vwap.pop(0)

            if len(self.KELP_prices) > timespan:
                self.KELP_prices.pop(0)

            fair_value = sum([x["vwap"]*x['vol'] for x in self.KELP_vwap]) / sum([x['vol'] for x in self.KELP_vwap])

            fair_value = mmmid_price

            # take all orders we can
            # for ask in order_depth.sell_orders.keys():
            #     if ask <= fair_value - KELP_take_width:
            #         ask_amount = -1 * order_depth.sell_orders[ask]
            #         if ask_amount <= 20:
            #             quantity = min(ask_amount, position_limit - position)
            #             if quantity > 0:
            #                 orders.append(Order("KELP", ask, quantity))
            #                 buy_order_volume += quantity

            # for bid in order_depth.buy_orders.keys():
            #     if bid >= fair_value + KELP_take_width:
            #         bid_amount = order_depth.buy_orders[bid]
            #         if bid_amount <= 20:
            #             quantity = min(bid_amount, position_limit + position)
            #             if quantity > 0:
            #                 orders.append(Order("KELP", bid, -1 * quantity))
            #                 sell_order_volume += quantity

            # only taking best bid/ask

            if best_ask <= fair_value - KELP_take_width:
                ask_amount = -1 * order_depth.sell_orders[best_ask]
                if ask_amount <= 20:
                    quantity = min(ask_amount, position_limit - position)
                    if quantity > 0:
                        orders.append(Order("KELP", best_ask, quantity))
                        buy_order_volume += quantity
            if best_bid >= fair_value + KELP_take_width:
                bid_amount = order_depth.buy_orders[best_bid]
                if bid_amount <= 20:
                    quantity = min(bid_amount, position_limit + position)
                    if quantity > 0:
                        orders.append(Order("KELP", best_bid, -1 * quantity))
                        sell_order_volume += quantity

            buy_order_volume, sell_order_volume = self.clear_position_order(orders, order_depth, position, position_limit, "KELP", buy_order_volume, sell_order_volume, fair_value, 2)

            aaf = [price for price in order_depth.sell_orders.keys() if price > fair_value + 1]
            bbf = [price for price in order_depth.buy_orders.keys() if price < fair_value - 1]
            baaf = min(aaf) if len(aaf) > 0 else fair_value + 2
            bbbf = max(bbf) if len(bbf) > 0 else fair_value - 2

            buy_quantity = position_limit - (position + buy_order_volume)
            if buy_quantity > 0:
                orders.append(Order("KELP", bbbf + 1, buy_quantity))  # Buy order

            sell_quantity = position_limit + (position - sell_order_volume)
            if sell_quantity > 0:
                orders.append(Order("KELP", baaf - 1, -sell_quantity))  # Sell order

        return orders

    def run(self, state: TradingState):
        result = {}

        rfr_fair_value = 10000  # Participant should calculate this value
        rfr_width = 2
        rfr_position_limit = 50

        KELP_make_width = 3.5
        KELP_take_width = 1
        KELP_position_limit = 50
        KELP_timemspan = 10

        # traderData = jsonpickle.decode(state.traderData)
        # print(state.traderData)
        # self.KELP_prices = traderData["KELP_prices"]
        # self.KELP_vwap = traderData["KELP_vwap"]

        if "RAINFOREST_RESIN" in state.order_depths:
            rfr_position = state.position["RAINFOREST_RESIN"] if "RAINFOREST_RESIN" in state.position else 0
            rfr_orders = self.rfr_orders(state.order_depths["RAINFOREST_RESIN"], rfr_fair_value, rfr_width, rfr_position, rfr_position_limit)
            result["RAINFOREST_RESIN"] = rfr_orders

        if "KELP" in state.order_depths:
            KELP_position = state.position["KELP"] if "KELP" in state.position else 0
            KELP_orders = self.KELP_orders(state.order_depths["KELP"], KELP_timemspan, KELP_make_width, KELP_take_width, KELP_position, KELP_position_limit)
            result["KELP"] = KELP_orders


        traderData = jsonpickle.encode( { "KELP_prices": self.KELP_prices, "KELP_vwap": self.KELP_vwap })


        conversions = 1

        return result, conversions, traderData
