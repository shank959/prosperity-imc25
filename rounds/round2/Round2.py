from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import string
import jsonpickle
import numpy as np
import math
import statistics


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
        self.PIC1 = {
            "position_limit": 60,
            "mPrice": [],
            "basket_weights": {
                "CROISSANTS": 6,
                "JAMS": 3,
                "DJEMBES": 1
            }
        }
        self.PIC2 = {
            "position_limit": 100,
            "mPrice": [],
            "basket_weights": {
                "CROISSANTS": 4,
                "JAMS": 2,
                "DJEMBES": 0
            }
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
    

    def garch_forecast(self, returns: List[float], omega: float, alpha: float, beta: float) -> np.ndarray:

        variance = np.zeros(len(returns))
        variance[0] = statistics.pvariance(returns)
        for t in range(1, len(returns)):
            variance[t] = omega + alpha * returns[t - 1]**2 + beta * variance[t - 1]
        volatility = np.sqrt(variance)
            
        return volatility
    

    def garch_log_likelihood(self, returns: List[float], omega: float, alpha: float, beta: float) -> float:

        volatility = self.garch_forecast(returns, omega, alpha, beta)
        log_likelihood = 0.0
        T = len(returns)
        # Compute the log-likelihood assuming normally distributed returns
        for t in range(T):
            log_likelihood += -0.5 * (np.log(2 * np.pi) + np.log(volatility[t]) + (returns[t]**2 / volatility[t]))
        return log_likelihood


    def rfr_orders(self, orderDep: OrderDepth, fairVal: int, width: int, position: int, positionLimit: int) -> List[Order]:
        orders: List[Order] = []

        buyVolume = 0
        sellVolume = 0

        baaf = min(
            [price for price in orderDep.sell_orders.keys() if price > fairVal + 1])
        bbbf = max([price for price in orderDep.buy_orders.keys()
                   if price < fairVal - 1])
        
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
        #         bestBid = max([price for price in orderDep.buy_orders.keys() if abs(orderDep.buy_orders[price]) >= min_vol])
        #         return (bestAsk + bestBid) / 2

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
    

    def pic1_order(self, order_depth: OrderDepth, position: int, position_limit: int):
        # pic1 and underlying


        pass

    def create_synthetic_basket_order_depth(self, order_depths, basket_weights):
        # create a synthetic basket order depth from a dictionary of order depths and basket weights
        synthetic_depth = OrderDepth()
        # get the best ask and best bid for each product in the basket
        best_CROISSANTS_ask = min(order_depths["CROISSANTS"].sell_orders.keys()) if order_depths["CROISSANTS"].sell_orders else float("inf")
        best_CROISSANTS_bid = max(order_depths["CROISSANTS"].buy_orders.keys()) if order_depths["CROISSANTS"].buy_orders else 0
        best_JAMS_ask = min(order_depths["JAMS"].sell_orders.keys()) if order_depths["JAMS"].sell_orders else float("inf")
        best_JAMS_bid = max(order_depths["JAMS"].buy_orders.keys()) if order_depths["JAMS"].buy_orders else 0
        if basket_weights["DJEMBES"] > 0:
            best_DJEMBES_ask = min(order_depths["DJEMBES"].sell_orders.keys()) if order_depths["DJEMBES"].sell_orders else float("inf")
            best_DJEMBES_bid = max(order_depths["DJEMBES"].buy_orders.keys()) if order_depths["DJEMBES"].buy_orders else 0
        else:
            best_DJEMBES_ask = float("inf")
            best_DJEMBES_bid = 0

        # get the implied best bid and best ask for the synthetic basket
        synthetic_bid = (
            best_CROISSANTS_bid * basket_weights["CROISSANTS"] +
            best_JAMS_bid * basket_weights["JAMS"] +
            best_DJEMBES_bid * basket_weights["DJEMBES"]
        )
        synthetic_ask = (
            best_CROISSANTS_ask * basket_weights["CROISSANTS"] +
            best_JAMS_ask * basket_weights["JAMS"] +
            best_DJEMBES_ask * basket_weights["DJEMBES"]
        )

        # compute available volumes - (use integer division by weights)
        if best_CROISSANTS_bid > 0 and best_JAMS_bid > 0 and (basket_weights["DJEMBES"] == 0 or best_DJEMBES_bid > 0):
            vol_CROISSANTS = order_depths["CROISSANTS"].buy_orders[best_CROISSANTS_bid] // basket_weights["CROISSANTS"]
            vol_JAMS = order_depths["JAMS"].buy_orders[best_JAMS_bid] // basket_weights["JAMS"]
            if basket_weights["DJEMBES"] > 0:
                vol_DJEMBES = order_depths["DJEMBES"].buy_orders[best_DJEMBES_bid] // basket_weights["DJEMBES"]
            else:
                vol_DJEMBES = float('inf')
            synthetic_bid_volume = min(vol_CROISSANTS, vol_JAMS, vol_DJEMBES)
            synthetic_depth.buy_orders[synthetic_bid] = synthetic_bid_volume

        if synthetic_ask < float("inf"):
            vol_CROISSANTS = -order_depths["CROISSANTS"].sell_orders[best_CROISSANTS_ask] // basket_weights["CROISSANTS"]
            vol_JAMS = -order_depths["JAMS"].sell_orders[best_JAMS_ask] // basket_weights["JAMS"]
            if basket_weights["DJEMBES"] > 0:
                vol_DJEMBES = -order_depths["DJEMBES"].sell_orders[best_DJEMBES_ask] // basket_weights["DJEMBES"]
            else:
                vol_DJEMBES = float('inf')
            synthetic_ask_volume = min(vol_CROISSANTS, vol_JAMS, vol_DJEMBES)
            synthetic_depth.sell_orders[synthetic_ask] = -synthetic_ask_volume

        return synthetic_depth
    

    def pic2_spread_order(self, order_depth: OrderDepth, position: int, position_limit: int):
        # pic2 and underlying
        # synthesize a basket order depth
        orders = []
        pic_order_depth = order_depth["PICNIC_BASKET2"]
        underlying_order_depths = {
            "CROISSANTS": order_depth["CROISSANTS"],
            "JAMS": order_depth["JAMS"],
            "DJEMBES": order_depth["DJEMBES"]
        }

        basket_weights = self.PIC2["basket_weights"]
        synthetic_depth = self.create_synthetic_basket_order_depth(underlying_order_depths, basket_weights)
        synthetic_bid = max(synthetic_depth.buy_orders.keys()) if synthetic_depth.buy_orders else None
        synthetic_ask = min(synthetic_depth.sell_orders.keys()) if synthetic_depth.sell_orders else None
        actual_bid = max(pic_order_depth.buy_orders.keys()) if pic_order_depth.buy_orders else None
        actual_ask = min(pic_order_depth.sell_orders.keys()) if pic_order_depth.sell_orders else None

        # Case 1: Basket undervalued relative to its synthetic value
        # Actual ask is lower than synthetic bid: Buy basket and sell underlying.
        if actual_ask is not None and synthetic_bid is not None and actual_ask < synthetic_bid:
            pic_buy_volume = abs(pic_order_depth.sell_orders[actual_ask])
            synthetic_sell_volume = synthetic_depth.buy_orders[synthetic_bid]
            exec_vol = min(pic_buy_volume, synthetic_sell_volume, position_limit - position)
            if exec_vol > 0:
                orders.append(Order("PICNIC_BASKET2", actual_ask, exec_vol))
                # Sell the underlying components at their best bid prices.
                for comp, weight in basket_weights.items():
                    if weight == 0:
                        continue
                    if comp in underlying_order_depths:
                        comp_depth = underlying_order_depths[comp]
                        best_bid = max(comp_depth.buy_orders.keys()) if comp_depth.buy_orders else None
                        if best_bid is not None:
                            orders.append(Order(comp, best_bid, -exec_vol * weight))

        # Case 2: Basket overvalued relative to its synthetic value
        # Actual bid is higher than synthetic ask: Sell basket and buy underlying.
        if actual_bid is not None and synthetic_ask is not None and actual_bid > synthetic_ask:
            pic_sell_volume = pic_order_depth.buy_orders[actual_bid]
            synthetic_buy_volume = abs(synthetic_depth.sell_orders[synthetic_ask])
            # Use the current position to limit the sell order (only sell if in a positive position)
            exec_vol = min(pic_sell_volume, synthetic_buy_volume, position if position > 0 else 0)
            if exec_vol > 0:
                orders.append(Order("PICNIC_BASKET2", actual_bid, -exec_vol))
                # Buy the underlying components at their best ask prices.
                for comp, weight in basket_weights.items():
                    if weight == 0:
                        continue
                    if comp in underlying_order_depths:
                        comp_depth = underlying_order_depths[comp]
                        best_ask = min(comp_depth.sell_orders.keys()) if comp_depth.sell_orders else None
                        if best_ask is not None:
                            orders.append(Order(comp, best_ask, exec_vol * weight))
        return orders

        

    def cross_pic_order(self, order_depth: OrderDepth, position: int, position_limit: int):
        # pic1 and pic2
        pass

    def run(self, state: TradingState):
        result = {}

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

        trader_object = {}
        if state.traderData:
            trader_object = jsonpickle.decode(state.traderData)

        if "initialized" not in trader_object:
            # First run — seed with historical fair values (replace with your real values)
            trader_object["squidfair_value_history"] = [1842.5, 1844.5, 1843.5, 1842.5, 1842.0, 1841.5, 1841.0, 1839.0, 1833.0,
                                                   1833.5, 1832.5, 1831.5, 1831.5, 1832.5, 1830.5, 1831.5, 1833.0, 1834.5, 1838.0, 1839.5
                                                   ]
            trader_object["initialized"] = True

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

        if "SQUID_INK" in state.order_depths:
            squid_position = state.position["SQUID_INK"] if "SQUID_INK" in state.position else 0
            squid_orders, squid_fair_value_history = self.SQUID_orders(
                state.order_depths["SQUID_INK"],
                squid_timespan,
                squid_take_width,          
                squid_position,   
                squid_position_limit,
                trader_object["squidfair_value_history"]
            )
        
        if "PICNIC_BASKET2" in state.order_depths:
            pic2_position = state.position["PICNIC_BASKET2"] if "PICNIC_BASKET2" in state.position else 0
            croissants_position = state.position["CROISSANTS"] if "CROISSANTS" in state.position else 0
            jams_position = state.position["JAMS"] if "JAMS" in state.position else 0
            
            pic2_orders, croissants_orders, jams_orders = self.pic2_spread_order(
                state.order_depths,
                pic2_position,
                croissants_position,
                jams_position,
                self.PIC2["position_limit"],
                self.CROISSANTS["position_limit"],
                self.JAMS["position_limit"]
            )
            
            result["PICNIC_BASKET2"] = pic2_orders
            result["CROISSANTS"] = croissants_orders 
            result["JAMS"] = jams_orders
            


        traderData = jsonpickle.encode({
            "priceKelp": self.KELP["mPrice"],
            "vwapKelp": self.KELP["vwap"],
            "fair_value_history": squid_fair_value_history,
            # "SQUID_prices": self.SQUID["mPrice"],
            # "SQUID_vwap": self.SQUID["vwap"]
        })

        conversions = 1
        return result, conversions, traderData