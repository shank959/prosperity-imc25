from datamodel import OrderDepth, UserId, TradingState, Order, Observation, ConversionObservation
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
        self.RFR = {
            "position_limit": 50,
            "fair_value": 10000,
        }
        self.KELP = {
            "position_limit": 50,
            "market_makers_size": 15,
            "mPrice": [],
            "vwap": []
        }
        self.SQUID = {
            "position_limit": 50,
            "market_makers_size": 15,
            "mPrice": [],
            "vwap": []
        }
        self.MACARONS = {
            "position_limit": 75,
            "conversion_limit": 10,
            "storage_cost": 0.1,
            "critical_sunlight_index": 44.95,
            "sunlight_data": [],
            "mean_price": 630,
            "sell_band": 670,
            "buy_band": 590,
        }
        self.BASKET1 = {              
            "premium": 50,
            "position_limit": 60,
            "trig_up":  85,
            "trig_dn": -85,
            "step":     60,  
        }
        self.BASKET2 = { 
            "premium": 40,
            "position_limit": 100,
            "trig_up":    90,
            "trig_dn":   -105,
            "stop_loss": -160,
            "step":       20,
            "exit_band":  10
        }
        self.RECIPE = {
            "PICNIC_BASKET1": {"CROISSANTS": 6, "JAMS": 3, "DJEMBES": 1},
            "PICNIC_BASKET2": {"CROISSANTS": 4, "JAMS": 2}
        }
        self.CROISSANTS = {
            "position_limit": 250,
            "mPrice": []
        }
        self.JAMS = {
            "position_limit": 350,
            "mPrice": []
        }
        self.DJEMBES = {
            "position_limit": 60,
            "mPrice": []
        }
        self.VOLCANIC_ROCK = {
            "position_limit": 400
        }
        self.VOLCANIC_ROCK_VOUCHER = {
            "position_limit": 200
        }

    def get_mm_price(self, order_depth: OrderDepth, product: str) -> float:

        # Find largest volume ask order
        max_ask_volume = max(abs(volume) for volume in order_depth.sell_orders.values())
        max_ask_price = [price for price, volume in order_depth.sell_orders.items() 
                        if abs(volume) == max_ask_volume][0]

        # Find largest volume bid order 
        max_bid_volume = max(abs(volume) for volume in order_depth.buy_orders.values())
        max_bid_price = [price for price, volume in order_depth.buy_orders.items()
                        if abs(volume) == max_bid_volume][0]

        # Calculate midprice
        return (max_ask_price + max_bid_price) / 2.0



    def rfr_orders(self, orderDep: OrderDepth, fairVal: int, width: int, position: int, positionLimit: int) -> List[Order]:
        orders: List[Order] = []

        buyVolume = 0
        sellVolume = 0


        asks = [p for p in orderDep.sell_orders if p > fairVal + 1]
        if asks:
            baaf = min(asks)
        else:
            baaf = fairVal + 1

        bids = [p for p in orderDep.buy_orders if p < fairVal -1]
        if bids:
            bbbf = max(bids)
        else:
            bbbf = fairVal - 1

        
        if len(orderDep.sell_orders) != 0:
            bestAsk = min(orderDep.sell_orders.keys())
            best_ask_amount = -1*orderDep.sell_orders[bestAsk]
            if bestAsk < fairVal:
                quantity = min(best_ask_amount, positionLimit -
                               position)  # max amt to buy
                if quantity > 0:
                    orders.append(Order("RAINFOREST_RESIN", round(bestAsk), quantity))
                    buyVolume += quantity

        if len(orderDep.buy_orders) != 0:
            bestBid = max(orderDep.buy_orders.keys())
            best_bid_amount = orderDep.buy_orders[bestBid]
            if bestBid > fairVal:
                # should be the max we can sell
                quantity = min(best_bid_amount, positionLimit + position)
                if quantity > 0:
                    orders.append(Order("RAINFOREST_RESIN", round(bestBid), -1 * quantity))
                    sellVolume += quantity

        buyVolume, sellVolume = self.clearPos(
            orders, orderDep, position, positionLimit, "RAINFOREST_RESIN", buyVolume, sellVolume, fairVal, 1)

        buyQuant = positionLimit - (position + buyVolume)
        if buyQuant > 0:
            orders.append(Order("RAINFOREST_RESIN", round(bbbf + 1), buyQuant))

        sellQuant = positionLimit + (position - sellVolume)
        if sellQuant > 0:
            orders.append(Order("RAINFOREST_RESIN", round(baaf - 1), -sellQuant))

        return orders

    def clearPos(self, orders: List[Order], orderDep: OrderDepth, position: int, positionLimit: int, product: str, buyVolume: int, sellVolume: int, fairVal: float, width: int) -> List[Order]:
        positionAfter = position + buyVolume - sellVolume
        fairBid = math.floor(fairVal)
        fairAsk = math.ceil(fairVal)

        buyQuant = positionLimit - (position + buyVolume)
        sellQuant = positionLimit + (position - sellVolume)

        if positionAfter > 0:
            if fairAsk in orderDep.buy_orders.keys():
                clearQuant = min(
                    orderDep.buy_orders[fairAsk], positionAfter)
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

    # Method: midPrice,
    def kelpfairValue(self, orderDep: OrderDepth, method="midPrice", min_vol=0) -> float:
        if method == "midPrice":
            bestAsk = min(orderDep.sell_orders.keys())
            bestBid = max(orderDep.buy_orders.keys())
            midPrice = (bestAsk + bestBid) / 2
            return midPrice
        #! TODO: TEST OUT DIFFERENT METHODS OF CALCULATING FAIR VALUE MM VWAP ECT

        # elif method == "midpriceFilter":
        #     if len([price for price in orderDep.sell_orders.keys() if abs(orderDep.sell_orders[price]) >= min_vol]) == 0 or len([price for price in orderDep.buy_orders.keys() if abs(orderDep.buy_orders[price]) >= min_vol]) == 0:
        #         bestAsk = min(orderDep.sell_orders.keys())
        #         bestBid = max(orderDep.buy_orders.keys())
        #         return (bestAsk + bestBid) / 2
        #     else:
        #         bestAsk = min([price for price in orderDep.sell_orders.keys() if abs(orderDep.sell_orders[price]) >= min_vol])

    def kelpOrders(self, orderDep: OrderDepth, timespan: int, width: float, kelpWidth: float, position: int, positionLimit: int) -> List[Order]:
        orders: List[Order] = []

        buyVolume = 0
        sellVolume = 0

        if len(orderDep.sell_orders) != 0 and len(orderDep.buy_orders) != 0:
            bestAsk = min(orderDep.sell_orders.keys())
            bestBid = max(orderDep.buy_orders.keys())
            filtered_ask = [price for price in orderDep.sell_orders.keys() if abs(orderDep.sell_orders[price]) >= 15]
            filtered_bid = [price for price in orderDep.buy_orders.keys() if abs(orderDep.buy_orders[price]) >= 15]
            mm_ask = min(filtered_ask) if filtered_ask else bestAsk
            mm_bid = max(filtered_bid) if filtered_bid else bestBid

            mmmid_price = (mm_ask + mm_bid) / 2
            self.KELP["mPrice"].append(mmmid_price)

            volume = -1 * \
                orderDep.sell_orders[bestAsk] + \
                orderDep.buy_orders[bestBid]
            vwap = (bestBid * (-1) * orderDep.sell_orders[bestAsk] +
                    bestAsk * orderDep.buy_orders[bestBid]) / volume
            self.KELP["vwap"].append({"vol": volume, "vwap": vwap})

            if len(self.KELP["vwap"]) > timespan:
                self.KELP["vwap"].pop(0)

            if len(self.KELP["mPrice"]) > timespan:
                self.KELP["mPrice"].pop(0)

            fairVal = mmmid_price

            if bestAsk <= fairVal - kelpWidth:
                ask_amount = -1 * orderDep.sell_orders[bestAsk]
                if ask_amount <= 20:
                    quantity = min(ask_amount, positionLimit - position)
                    if quantity > 0:
                        orders.append(Order("KELP", round(bestAsk), quantity))
                        buyVolume += quantity

            if bestBid >= fairVal + kelpWidth:
                bid_amount = orderDep.buy_orders[bestBid]
                if bid_amount <= 20:
                    quantity = min(bid_amount, positionLimit + position)
                    if quantity > 0:
                        orders.append(Order("KELP", round(bestBid), -1 * quantity))
                        sellVolume += quantity

            buyVolume, sellVolume = self.clearPos(
                orders, orderDep, position, positionLimit, "KELP", buyVolume, sellVolume, fairVal, 2)

            #! MARKET MAKING, TRY NEW METHODS with different thresholders

            aaf = [price for price in orderDep.sell_orders.keys() if price > fairVal + 1]
            bbf = [price for price in orderDep.buy_orders.keys() if price < fairVal - 1]
            baaf = min(aaf) if aaf else fairVal + 2
            bbbf = max(bbf) if bbf else fairVal - 2

            buyQuant = positionLimit - (position + buyVolume)
            if buyQuant > 0:
                orders.append(Order("KELP", round(bbbf + 1), buyQuant))

            sellQuant = positionLimit + (position - sellVolume)
            if sellQuant > 0:
                orders.append(Order("KELP", round(baaf - 1), -sellQuant))

        return orders

    # def SQUID_orders(self, orderDep: OrderDepth, timespan: int, make_width: float, take_width: float, position: int, positionLimit: int) -> List[Order]:
    #     orders: List[Order] = []

    #     if not orderDep.sell_orders or not orderDep.buy_orders:
    #         return orders

    #     bestAsk = min(orderDep.sell_orders.keys())
    #     bestBid = max(orderDep.buy_orders.keys())
    #     midPrice = (bestAsk + bestBid) / 2.0

    #     offset = 1.0  # ! TODO: adjust

    #     if position < positionLimit:
    #         buy_qty = positionLimit - position
    #         orders.append(Order("SQUID_INK", round(midPrice - offset), buy_qty))

    #     if position > -positionLimit:
    #         sell_qty = positionLimit + position
    #         orders.append(Order("SQUID_INK", round(midPrice + offset), -sell_qty))

    #     return orders

    # def squid_ink_fetch_historical_data(self):
    #     day_price_0_df = pd.read_csv(
    #         "../round-1-island-data-bottle/prices_round_1_day_0.csv", delimiter=";")
    #     day_price_m1_df = pd.read_csv(
    #         "../round-1-island-data-bottle/prices_round_1_day_-1.csv", delimiter=";")
    #     day_price_m2_df = pd.read_csv(
    #         "../round-1-island-data-bottle/prices_round_1_day_-2.csv", delimiter=";")

    #     merged_df = pd.concat(
    #         [day_price_0_df, day_price_m1_df, day_price_m2_df])

    #     squid_ink_df = merged_df[merged_df["product"] == "SQUID_INK"].copy()
    #     return squid_ink_df

    # def squid_ink_mean_reversion(self):
    #     # We want to combine this strategy with others to do directional trading at specific points
    #     z_score_threshold = 1.5  # TODO: tune hyperparameter

    #     squid_ink_df = self.squid_ink_fetch_historical_data()

    #     mean_price = squid_ink_df['midPrice'].mean()
    #     std_price = squid_ink_df['midPrice'].std()

    #     if std_price == 0:
    #         squid_ink_df['z_score'] = 0
    #     else:
    #         squid_ink_df['z_score'] = (
    #             squid_ink_df['midPrice'] - mean_price) / std_price

    #     squid_ink_df['mr_signal'] = 0  # 0 = hold, 1 = buy, -1 = sell
    #     squid_ink_df.loc[squid_ink_df['z_score'] < -
    #                      z_score_threshold, 'mr_signal'] = 1  # Buy signal
    #     squid_ink_df.loc[squid_ink_df['z_score'] >
    #                      z_score_threshold, 'mr_signal'] = -1  # Sell signal


    #     return squid_ink_df

    # def momentum_strategy(self, lookback=1):
    #     lookback = 3  # TODO: tune hyperparameter
    #     squid_ink_df = self.squid_ink_fetch_historical_data()
    #     squid_ink_df['price_change'] = squid_ink_df['midPrice'].diff(lookback)
    #     squid_ink_df['momentum_signal'] = 0
    #     squid_ink_df.loc[squid_ink_df['price_change']
    #                      > 0, 'momentum_signal'] = 1  # Buy signal
    #     squid_ink_df.loc[squid_ink_df['price_change']
    #                      < 0, 'momentum_signal'] = -1  # Sell signal

    #     return squid_ink_df

    def SQUID_orders(self, order_depth: OrderDepth, timespan: int, take_width: float, position: int, position_limit: int, fair_value_history: List[float]) -> List[Order]:
        orders: List[Order] = []

        buy_order_volume = 0
        sell_order_volume = 0

        if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            filtered_ask = [price for price in order_depth.sell_orders.keys() if abs(order_depth.sell_orders[price]) >= 15]
            filtered_bid = [price for price in order_depth.buy_orders.keys() if abs(order_depth.buy_orders[price]) >= 15]
            mm_ask = min(filtered_ask) if filtered_ask else best_ask
            mm_bid = max(filtered_bid) if filtered_bid else best_bid

            mmmid_price = (mm_ask + mm_bid) / 2
            self.SQUID["mPrice"].append(mmmid_price)

            volume = -1 * order_depth.sell_orders[best_ask] + order_depth.buy_orders[best_bid]
            vwap = (best_bid * (-1) * order_depth.sell_orders[best_ask] +
                    best_ask * order_depth.buy_orders[best_bid]) / volume
            self.SQUID["vwap"].append({"vol": volume, "vwap": vwap})

            if len(self.SQUID["vwap"]) > timespan:
                self.SQUID["vwap"].pop(0)

            if len(self.SQUID["mPrice"]) > timespan:
                self.SQUID["mPrice"].pop(0)

            fair_value = mmmid_price  # You can also use VWAP or rolling mean here

            # moving average of fair value history
            fair_value_history.append(fair_value)
            if len(fair_value_history) > timespan:
                fair_value_history.pop(0)
            fair_value_mean = sum(fair_value_history) / len(fair_value_history)

            # volatility calculation
            vol = np.std(fair_value_history)

            # === Z-score calculation ===
            z = (fair_value - fair_value_mean) / vol if vol > 0 else 0
            z_threshold = 1.5  # adjustable threshold

            # === Modified trading logic ===
            if z < -z_threshold and best_ask <= fair_value - take_width:
                ask_amount = -order_depth.sell_orders[best_ask]
                if ask_amount <= 20:
                    quantity = min(ask_amount, position_limit - position)
                    if quantity > 0:
                        orders.append(Order("SQUID_INK", round(best_ask), quantity))
                        buy_order_volume += quantity

            if z > z_threshold and best_bid >= fair_value + take_width:
                bid_amount = order_depth.buy_orders[best_bid]
                if bid_amount <= 20:
                    quantity = min(bid_amount, position_limit + position)
                    if quantity > 0:
                        orders.append(Order("SQUID_INK", round(best_bid), -quantity))
                        sell_order_volume += quantity

            # if best_ask <= fair_value - take_width:
            #     ask_amount = -1 * order_depth.sell_orders[best_ask]
            #     if ask_amount <= 20:
            #         quantity = min(ask_amount, position_limit - position)
            #         if quantity > 0:
            #             orders.append(Order("SQUID_INK", round(best_ask), quantity))
            #             buy_order_volume += quantity

            # if best_bid >= fair_value + take_width:
            #     bid_amount = order_depth.buy_orders[best_bid]
            #     if bid_amount <= 20:
            #         quantity = min(bid_amount, position_limit + position)
            #         if quantity > 0:
            #             orders.append(Order("SQUID_INK", round(best_bid), -1 * quantity))
                        # sell_order_volume += quantity

            buy_order_volume, sell_order_volume = self.clearPos(
                orders, order_depth, position, position_limit, "SQUID_INK", buy_order_volume, sell_order_volume, fair_value, 2
            )

            # aaf = [price for price in order_depth.sell_orders.keys() if price > fair_value + 1]
            # bbf = [price for price in order_depth.buy_orders.keys() if price < fair_value - 1]
            # baaf = min(aaf) if aaf else fair_value + 2
            # bbbf = max(bbf) if bbf else fair_value - 2

            # === Z-score skewed Market Making ===
            mid_price = (best_ask + best_bid) / 2
            z = (mid_price - fair_value) / vol if vol > 0 else 0
            z_threshold = 1.5  # adjustable threshold
            skew_distance = 1  # base quote distance

            # Adjust quotes based on z-score
            if z > z_threshold:
                # Expect price drop: quote closer ask, farther bid
                ask_price = round(fair_value + skew_distance)
                bid_price = round(fair_value - (skew_distance + abs(z)))

            elif z < -z_threshold:
                # Expect price rise: quote closer bid, farther ask
                ask_price = round(fair_value + (skew_distance + abs(z)))
                bid_price = round(fair_value - skew_distance)

            else:
                # No strong signal: symmetric quotes
                ask_price = round(fair_value + skew_distance)
                bid_price = round(fair_value - skew_distance)

            # Calculate quantities to remain within position limits
            buy_quantity = position_limit - (position + buy_order_volume)
            sell_quantity = position_limit + (position - sell_order_volume)

            if buy_quantity > 0:
                orders.append(Order("SQUID_INK", bid_price, buy_quantity))

            if sell_quantity > 0:
                orders.append(Order("SQUID_INK", ask_price, -sell_quantity))

            # buy_quantity = position_limit - (position + buy_order_volume)
            # if buy_quantity > 0:
            #     orders.append(Order("SQUID_INK", round(bbbf + 1), buy_quantity))

            # sell_quantity = position_limit + (position - sell_order_volume)
            # if sell_quantity > 0:
            #     orders.append(Order("SQUID_INK", round(baaf - 1), -sell_quantity))

        return orders, fair_value_history

    def macaron_get_sunlight_state(self, observation: ConversionObservation) -> Dict[str, Any]:
        if len(self.MACARONS["sunlight_data"]) > 2:
            roc = (observation.sunlightIndex - self.MACARONS["sunlight_data"][-2]) / self.MACARONS["sunlight_data"][-2]
        else:
            roc = 0
        return {
            "is_panic_mode": observation.sunlightIndex <= self.MACARONS["critical_sunlight_index"], 
            "sunlight_index": observation.sunlightIndex,
            "delta": self.MACARONS["critical_sunlight_index"] - observation.sunlightIndex,
            "roc": roc
        }
    
    # def get_sunlight_data(self, observation: ConversionObservation) -> float:
    #     current_sunlight_index = observation.sunlightIndex
    #     self.MACARONS["sunlight_data_window"].append(current_sunlight_index)

    def deepening_panic_mode(self,
                        orderDep: OrderDepth,
                        observation: ConversionObservation,
                        position: int) -> List[Order]:
        orders: List[Order] = []
        state = self.macaron_get_sunlight_state(observation)

        # 1) Compute aggressiveness exactly as before
        critical = self.MACARONS["critical_sunlight_index"]
        delta    = state["delta"]
        roc      = state["roc"] # should be negative in this function
        cap      = self.MACARONS["position_limit"] - position

        # 3) Grab the *local* best ask/order levels
        asks = sorted(orderDep.sell_orders.items())
        bids = sorted(orderDep.buy_orders.items(), reverse=True)
        best_ask = asks[0][0]
        best_bid = bids[0][0]

        # 4) Layered limit bids against the *local* asks
        remaining = cap
        for price, avail in asks:
            if remaining <= 0:
                break
            # Weight most toward the top of book
            layer_qty = round(min(avail, remaining * 0.5))
            orders.append(Order("MAGNIFICENT_MACARONS", price, layer_qty))
            remaining -= layer_qty

        # 6) Take‑profit sells around the *local* mid
        local_mid = (best_ask + best_bid) // 2
        # average spread is 8.5 and we wanna capture half but lets not be greedy
        profit_spread =  max(2, (best_ask - best_bid) // 2)
        exit_price   = local_mid + profit_spread
        exit_qty     = round(position * 0.2)
        orders.append(Order("MAGNIFICENT_MACARONS", exit_price, -exit_qty))

        return orders
    
    def lightening_panic_mode(self,
                            orderDep: OrderDepth,
                            observation: ConversionObservation,
                            position: int) -> List[Order]:
        # when roc is increasing, we need to sell more aggressively
        orders: List[Order] = []
        state = self.macaron_get_sunlight_state(observation)
        roc = state["roc"]
        delta = state["delta"]
        alpha = delta / roc if roc != 0 else float('inf') # timestamps until above critical
        threshold_TTM = 50000 # will go above critical in estimated 50000 timestamps #! HP
        if alpha < threshold_TTM and position > 0:
            # start aggressively selling
            bids = sorted(orderDep.buy_orders.items(), reverse=True)
            best_bid = bids[0][0]
            best_bid_volume = orderDep.buy_orders[best_bid]
            orders.append(Order("MAGNIFICENT_MACARONS", best_bid, -best_bid_volume))

        return orders

    def macaron_band_orders(self,
                            orderDep: OrderDepth,
                            position: int) -> List[Order]:
        """
        In non‑panic mode, trade strictly within the fixed bands:
          • sell into any ask in [680,720]
          • buy from any bid in [550,580]
        """
        m   = self.MACARONS
        sell_band = m["sell_band"]
        buy_band = m["buy_band"]
        cap_limit = m["position_limit"]

        orders = []
        buy_volume = 0
        sell_volume = 0
        depth = orderDep
        # SELL band
        best_bid = max(depth.buy_orders.keys())
        best_bid_volume = depth.buy_orders[best_bid]
        if best_bid > sell_band:
            orders.append(Order("MAGNIFICENT_MACARONS", best_bid, -best_bid_volume))
            sell_volume += best_bid_volume
        # BUY band
        best_ask = min(depth.sell_orders.keys())
        best_ask_volume = depth.sell_orders[best_ask]
        if best_ask < buy_band:
            orders.append(Order("MAGNIFICENT_MACARONS", best_ask, best_ask_volume))
            buy_volume += best_ask_volume

        return orders, buy_volume, sell_volume

    

    def implied_bid_ask_macarons(self, observation: ConversionObservation):
        """
        Calculate implied bid and ask prices for macaroons based on order depth and observation data.
        """
        return (observation.bidPrice - observation.transportFees - observation.exportTariff - self.MACARONS["storage_cost"]), (observation.askPrice + observation.transportFees + observation.importTariff)
        
        
    def macarons_take_orders(self, orderDep: OrderDepth, observation: ConversionObservation):
        """
        Take orders for macaroons based on implied bid/asks and observation data.
        """
        orders: List[Order] = []
        take_width = 2
        buy_order_volume = 0
        sell_order_volume = 0

        implied_bid, implied_ask = self.implied_bid_ask_macarons(observation)

        buy_quantity = 10
        sell_quantity = 10

        # 1) HITTING THE SELL ORDERS (we buy)
        for price in sorted(orderDep.sell_orders.keys()):
            if price > implied_bid - take_width:
                break
            qty = min(orderDep.sell_orders[price], buy_quantity)
            if qty > 0:
                orders.append(Order("MAGNIFICENT_MACARONS", price, qty))
                buy_order_volume += qty

        # 2) HITTING THE BUY ORDERS (we sell)
        for price in sorted(orderDep.buy_orders.keys(), reverse=True):
            if price < implied_ask + take_width:
                break
            qty = min(orderDep.buy_orders[price], sell_quantity)
            if qty > 0:
                orders.append(Order("MAGNIFICENT_MACARONS", price, -qty))
                sell_order_volume += qty

        return orders, buy_order_volume, sell_order_volume
    

    
    def macarons_conversions(self, position: int) -> int:
        conversions = -position if abs(position) <= 10 else 10
        return conversions 
    

    # ------------------------ BASKET TRADING ------------------------
    def _best_bid(self, depth: OrderDepth) -> int:
        return max(depth.buy_orders.keys())

    def _best_ask(self, depth: OrderDepth) -> int:
        return min(depth.sell_orders.keys())

    def _mid(self, depth: OrderDepth) -> float:
        return (self._best_bid(depth) + self._best_ask(depth)) / 2

    def _synthetic_mid(self, basket: str, depths: Dict[str, OrderDepth]) -> float:
        return sum(self._mid(depths[p]) * q for p, q in self.RECIPE[basket].items())
    
    def trade_basket1(self, depths: Dict[str, OrderDepth], pos: int) -> List[Order]:
        d       = depths["PICNIC_BASKET1"]
        s       = self._mid(d) - self._synthetic_mid("PICNIC_BASKET1", depths) - self.BASKET1["premium"]
        orders  : List[Order] = []
        limit   = self.BASKET1["position_limit"]

        # SELL if overpriced
        if s > self.BASKET1["trig_up"] and pos > -limit:
            qty = min(self.BASKET1["step"], limit + pos, d.buy_orders[self._best_bid(d)])
            if qty > 0:
                orders.append(Order("PICNIC_BASKET1", self._best_bid(d), -qty))

        # BUY if under‑priced
        if s < self.BASKET1["trig_dn"] and pos < limit:
            qty = min(self.BASKET1["step"], limit - pos, abs(d.sell_orders[self._best_ask(d)]))
            if qty > 0:
                orders.append(Order("PICNIC_BASKET1", self._best_ask(d), qty))

        return orders
    
    def trade_basket2(self, depths: Dict[str, OrderDepth], pos: int) -> List[Order]:
        d = depths["PICNIC_BASKET2"]
        s = self._mid(d) - self._synthetic_mid("PICNIC_BASKET2", depths) - self.BASKET2["premium"]
        orders : List[Order] = []
        limit  = self.BASKET2["position_limit"]

        # exit/flatten if back near baseline
        if abs(s) < self.BASKET2["exit_band"] and pos != 0:
            if pos > 0:
                qty = min(pos, d.buy_orders[self._best_bid(d)])
                orders.append(Order("PICNIC_BASKET2", self._best_bid(d), -qty))
            else:
                qty = min(-pos, abs(d.sell_orders[self._best_ask(d)]))
                orders.append(Order("PICNIC_BASKET2", self._best_ask(d), qty))
            return orders

        # SELL (overpriced)
        if s > self.BASKET2["trig_up"] and pos > -limit:
            qty = min(self.BASKET2["step"], limit + pos, d.buy_orders[self._best_bid(d)])
            if qty > 0:
                orders.append(Order("PICNIC_BASKET2", self._best_bid(d), -qty))

        # BUY (underpriced)
        if s < self.BASKET2["trig_dn"] and pos < limit:
            qty = min(self.BASKET2["step"], limit - pos, abs(d.sell_orders[self._best_ask(d)]))
            if qty > 0:
                orders.append(Order("PICNIC_BASKET2", self._best_ask(d), qty))

        # hard stop‑loss (!TODO: tes)
        if s < self.BASKET2["stop_loss"] and pos > 0:
            qty = min(pos, d.buy_orders[self._best_bid(d)])
            orders.append(Order("PICNIC_BASKET2", self._best_bid(d), -qty))

        return orders
    
#! VOLCANIC ROCK TRADING
    
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
    
    def midprice(self, product, order_depth: OrderDepth) -> float:
        best_bid = max(order_depth[product].buy_orders.keys())\
            if order_depth[product].buy_orders else 0
        best_ask = min(order_depth[product].sell_orders.keys())\
            if order_depth[product].sell_orders else 0
        return (best_bid + best_ask) / 2 if best_bid and best_ask\
            else best_bid + best_ask

    def volc_orders(self, order_depth: OrderDepth,
                    positions: Dict[str, int], timestamp: int,
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
        info["VOLCANIC_ROCK_VOUCHER_9750"]["critical_boundary"] = 0.002

        info["VOLCANIC_ROCK_VOUCHER_10000"]["upper_threshold"] = 0.00534
        info["VOLCANIC_ROCK_VOUCHER_10000"]["lower_threshold"] = -0.01640
        info["VOLCANIC_ROCK_VOUCHER_10000"]["critical_boundary"] = 0.002

        info["VOLCANIC_ROCK_VOUCHER_10250"]["upper_threshold"] = 0.00502
        info["VOLCANIC_ROCK_VOUCHER_10250"]["lower_threshold"] = -0.02194
        info["VOLCANIC_ROCK_VOUCHER_10250"]["critical_boundary"] = 0.002

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
        result = {}
        conversions = 0
        macaron_buy_order_volume = 0
        macaron_sell_order_volume = 0

        rfr_fair_value = 10000  # Participant should calculate this value
        rfr_width = 2
        rfr_position_limit = 50

        # ! TODO: CHANGE TO WIDTH BEING A FUNCTION OF DIFFERENCE WITH FAIR VALUE AND VOLATILITY
        KELP_make_width = 3.5
        # ! TODO: CHANGE TO WIDTH BEING A FUNCTION OF DIFFERENCE WITH FAIR VALUE AND VOLATILITY
        kelpWidth = 1
        KELP_position_limit = 50
        KELP_timemspan = 10

        squid_make_width = 3.0  # ! TODO: adjust
        squid_take_width = 1.0  # ! TODO: adjust
        squid_position_limit = 50
        squid_timespan = 10

        # traderData = jsonpickle.decode(state.traderData)
        # print(state.traderData)
        # self.KELP["mPrice"] = traderData["priceKelp"]
        # self.KELP["vwap"] = traderData["vwapKelp"]

        traderObject = {"macaron_position": 0}
        if state.traderData:
            traderObject = jsonpickle.decode(state.traderData)
            self.MACARONS["sunlight_data"] = traderObject["sunlight_data"]

        if "RAINFOREST_RESIN" in state.order_depths:
            rfr_position = state.position["RAINFOREST_RESIN"] if "RAINFOREST_RESIN" in state.position else 0
            rfr_orders = self.rfr_orders(
                state.order_depths["RAINFOREST_RESIN"], rfr_fair_value, rfr_width, rfr_position, rfr_position_limit)
            result["RAINFOREST_RESIN"] = rfr_orders

        if "KELP" in state.order_depths:
            KELP_position = state.position["KELP"] if "KELP" in state.position else 0
            kelpOrders = self.kelpOrders(
                state.order_depths["KELP"], KELP_timemspan, KELP_make_width, kelpWidth, KELP_position, KELP_position_limit)
            result["KELP"] = kelpOrders


        if state.own_trades.get("MAGNIFICENT_MACARONS", 0):
            print(f"own_trades: {state.own_trades.get('MAGNIFICENT_MACARONS', 0)}")


        if "MAGNIFICENT_MACARONS" in state.order_depths:
            # check if in panic mode
            observation = state.observations.conversionObservations["MAGNIFICENT_MACARONS"]
            sunlight_state = self.macaron_get_sunlight_state(observation)
            self.MACARONS["sunlight_data"].append(observation.sunlightIndex)
            if len(self.MACARONS["sunlight_data"]) > 2:
                self.MACARONS["sunlight_data"].pop(0)

            # # print all data
            # print(f"sunlight_index: {state.observations.conversionObservations['MAGNIFICENT_MACARONS'].sunlightIndex}")
            # print(f"best_bid: {sorted(state.order_depths['MAGNIFICENT_MACARONS'].buy_orders.keys(), reverse=True)}")
            # print(f"best_ask: {sorted(state.order_depths['MAGNIFICENT_MACARONS'].sell_orders.keys())}")
            # print(f"midprice: {(sorted(state.order_depths['MAGNIFICENT_MACARONS'].buy_orders.keys(), reverse=True)[0] + sorted(state.order_depths['MAGNIFICENT_MACARONS'].sell_orders.keys())[0]) / 2}")
            # implied_bid, implied_ask = self.implied_bid_ask_macarons(observation)
            # print(f"implied_bid: {implied_bid}")
            # print(f"implied_ask: {implied_ask}")


            if sunlight_state["is_panic_mode"]:
                if sunlight_state["roc"] <= 0:
                    orders = self.deepening_panic_mode(state.order_depths["MAGNIFICENT_MACARONS"], observation, traderObject["macaron_position"])
                else:
                    orders = self.lightening_panic_mode(state.order_depths["MAGNIFICENT_MACARONS"], observation, traderObject["macaron_position"])
                take_orders, macaron_buy_order_volume, macaron_sell_order_volume = self.macarons_take_orders(state.order_depths["MAGNIFICENT_MACARONS"], observation)
                orders = orders + take_orders
            else: # if not in panic mode, perform normal band trading, exchange arb and market making
                # band_orders, buy_order_volume, sell_order_volume = self.macaron_band_orders(state.order_depths["MAGNIFICENT_MACARONS"], traderObject["macaron_position"])
                take_orders, macaron_buy_order_volume, macaron_sell_order_volume = self.macarons_take_orders(state.order_depths["MAGNIFICENT_MACARONS"], observation)
                conversions = self.macarons_conversions(macaron_buy_order_volume - macaron_sell_order_volume)
                orders = take_orders
            result["MAGNIFICENT_MACARONS"] = orders
            print(state.position.get("MAGNIFICENT_MACARONS", 0))
            # buy_order_volume = 0
            # sell_order_volume = 0
            # for order in orders:
            #     if order.quantity > 0:
            #         buy_order_volume += order.quantity
            #     else:
            #         sell_order_volume += order.quantity

        # ------------------------ BASKET TRADING ------------------------        
        if "PICNIC_BASKET1" in state.order_depths:
            pos1 = state.position.get("PICNIC_BASKET1", 0)
            result["PICNIC_BASKET1"] = self.trade_basket1(state.order_depths, pos1)

        if "PICNIC_BASKET2" in state.order_depths:
            pos2 = state.position.get("PICNIC_BASKET2", 0)
            result["PICNIC_BASKET2"] = self.trade_basket2(state.order_depths, pos2)
        # ---------------------------------------------------------------
        # Generate orders
        if "VOLCANIC_ROCK" in state.order_depths:
            volc_orders = self.volc_orders(state.order_depths,
                                           state.position,
                                           state.timestamp,
                                           aggression=float("inf"))

            # TEST FOR 10500 CHEESE
            if "VOLCANIC_ROCK_VOUCHER_10500" in state.order_depths:
                if state.order_depths["VOLCANIC_ROCK_VOUCHER_10500"].buy_orders:
                    best_bid = max(state.order_depths["VOLCANIC_ROCK_VOUCHER_10500"].buy_orders.keys())
                    if best_bid >= 4:
                        volc_orders.append(Order("VOLCANIC_ROCK_VOUCHER_10500", best_bid, - 30))
                if state.order_depths["VOLCANIC_ROCK_VOUCHER_10500"].sell_orders:
                    best_ask = min(state.order_depths["VOLCANIC_ROCK_VOUCHER_10500"].sell_orders.keys())
                    if best_ask <= 2:
                        volc_orders.append(Order("VOLCANIC_ROCK_VOUCHER_10500", best_ask, 30))

            for order in volc_orders:
                result.setdefault(order.symbol, []).append(order)

        traderData = jsonpickle.encode({
            "macaron_position": traderObject["macaron_position"] + (macaron_buy_order_volume - macaron_sell_order_volume) + conversions,
            "sunlight_data": self.MACARONS["sunlight_data"],
            "priceKelp": self.KELP["mPrice"],
            "vwapKelp": self.KELP["vwap"],
        })

        return result, conversions, traderData