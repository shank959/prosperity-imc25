from prosperity3bt.datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import string
import jsonpickle
import numpy as np
import math
import statistics


#! TODO: CREATE FUNCTION TO FORECAST VOLATILITY FOR SQUID INK AND KELP AND CALL IN RUN METHOD
#! FIND PATTERNS IN SQUID INK PRICE OR VOLATILITY USING TIME SERIES ANALYSIS
class Trader:
    def __init__(self):
        self.priceKelp = []
        self.vwapKelp = []
        # self.SQUID_prices = []
        # self.SQUID_vwap = []
        # self.SQUID_INK_prices = []
        # self.SQUID_INK_volatility_history = []

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


    def grid_search(self):
        omega = 0.00001
        alpha = 0.1
        beta = 0.8

        # Convert price history to returns
        returns = []
        for i in range(1, len(self.SQUID_INK_prices)):
            returns.append((self.SQUID_INK_prices[i] - self.SQUID_INK_prices[i-1]) / self.SQUID_INK_prices[i-1])
        returns = np.array(returns)

        best_ll = -np.inf
        best_params = (None, None, None)
        # Create grids for parameter candidates
        for omega_candidate in np.linspace(1e-8, 1e-5, 5):
            for alpha_candidate in np.linspace(0.05, 0.15, 5):
                for beta_candidate in np.linspace(0.80, 0.95, 5):
                    # Ensure stationarity: alpha + beta < 1
                    if alpha_candidate + beta_candidate < 1:
                        ll = self.garch_log_likelihood(returns.tolist(), omega_candidate, alpha_candidate, beta_candidate)
                        if ll > best_ll:
                            best_ll = ll
                            best_params = (omega_candidate, alpha_candidate, beta_candidate)

        print("Optimal Parameters (omega, alpha, beta):", best_params)
        print("Optimal Log-Likelihood:", best_ll)



    def rfr_orders(self, orderDep: OrderDepth, fairVal: int, width: int, position: int, positionLimit: int) -> List[Order]:
        orders: List[Order] = []

        buyVolume = 0
        sellVolume = 0

        baaf = min(
            [price for price in orderDep.sell_orders.keys() if price > fairVal + 1])
        bbbf = max([price for price in orderDep.buy_orders.keys()
                   if price < fairVal - 1])
        #! TODO: TRY DIFFERENT HYPERPAREMETERS FOR MARKET MAKING (VOLOTAILITY CALCULATION)
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
        #! TODO: TRY OUT DIFFERENT METHODS OF OFLLOADING position
        # fairAsk = fairBid = fair

        buyQuant = positionLimit - (position + buyVolume)
        sellQuant = positionLimit + (position - sellVolume)

        if positionAfter > 0:
            if fairAsk in orderDep.buy_orders.keys():
                clearQuant = min(
                    orderDep.buy_orders[fairAsk], positionAfter)
                #! TODO: SEE IF WE WANT TO OFFLOAD ENTIRE POSITIONS
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

        elif method == "midpriceFilter":
            if len([price for price in orderDep.sell_orders.keys() if abs(orderDep.sell_orders[price]) >= min_vol]) == 0 or len([price for price in orderDep.buy_orders.keys() if abs(orderDep.buy_orders[price]) >= min_vol]) == 0:
                bestAsk = min(orderDep.sell_orders.keys())
                bestBid = max(orderDep.buy_orders.keys())
                return (bestAsk + bestBid) / 2
            else:
                bestAsk = min([price for price in orderDep.sell_orders.keys() if abs(orderDep.sell_orders[price]) >= min_vol])
                bestBid = max([price for price in orderDep.buy_orders.keys() if abs(orderDep.buy_orders[price]) >= min_vol])
                return (bestAsk + bestBid) / 2

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
            self.priceKelp.append(mmmid_price)

            volume = -1 * \
                orderDep.sell_orders[bestAsk] + \
                orderDep.buy_orders[bestBid]
            vwap = (bestBid * (-1) * orderDep.sell_orders[bestAsk] +
                    bestAsk * orderDep.buy_orders[bestBid]) / volume
            self.vwapKelp.append({"vol": volume, "vwap": vwap})

            if len(self.vwapKelp) > timespan:
                self.vwapKelp.pop(0)

            if len(self.priceKelp) > timespan:
                self.priceKelp.pop(0)

            fairVal = mmmid_price

            if bestAsk <= fairVal - kelpWidth:
                ask_amount = -1 * orderDep.sell_orders[bestAsk]
                if ask_amount <= 20:    #! WHY 20? WHY NOT 15 LIKE EARLIER? ARE WE TO ASSUME 20 IS OPTIMAL THRESDHOLD FOR POS EV? 
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

    def run(self, state: TradingState):
        result = {}

        rfr_fair_value = 10000  # Participant should calculate this value
        rfr_width = 2
        rfr_positionition_limit = 50

        # ! TODO: CHANGE TO WIDTH BEING A FUNCTION OF DIFFERENCE WITH FAIR VALUE AND VOLATILITY
        KELP_make_width = 3.5
        # ! TODO: CHANGE TO WIDTH BEING A FUNCTION OF DIFFERENCE WITH FAIR VALUE AND VOLATILITY
        kelpWidth = 1
        KELP_positionition_limit = 50
        KELP_timemspan = 10

        # squid_make_width = 3.0  # ! TODO: adjust
        # squid_take_width = 1.0  # ! TODO: adjust
        # squid_positionition_limit = 50
        # squid_timespan = 10

        # traderData = jsonpickle.decode(state.traderData)
        # print(state.traderData)
        # self.priceKelp = traderData["priceKelp"]
        # self.vwapKelp = traderData["vwapKelp"]

        if "RAINFOREST_RESIN" in state.order_depths:
            rfr_positionition = state.position["RAINFOREST_RESIN"] if "RAINFOREST_RESIN" in state.position else 0
            rfr_orders = self.rfr_orders(
                state.order_depths["RAINFOREST_RESIN"], rfr_fair_value, rfr_width, rfr_positionition, rfr_positionition_limit)
            result["RAINFOREST_RESIN"] = rfr_orders

        if "KELP" in state.order_depths:
            KELP_positionition = state.position["KELP"] if "KELP" in state.position else 0
            kelpOrders = self.kelpOrders(
                state.order_depths["KELP"], KELP_timemspan, KELP_make_width, kelpWidth, KELP_positionition, KELP_positionition_limit)
            result["KELP"] = kelpOrders

        # if "SQUID_INK" in state.order_depths:
        #     squid_positionition = state.position["SQUID_INK"] if "SQUID_INK" in state.position else 0
        #     squid_orders = self.SQUID_orders(
        #         state.order_depths["SQUID_INK"], squid_timespan, squid_make_width, squid_take_width, squid_positionition, squid_positionition_limit)
        #     result["SQUID_INK"] = squid_orders

        traderData = jsonpickle.encode({
            "priceKelp": self.priceKelp,
            "vwapKelp": self.vwapKelp,
            # "SQUID_prices": self.SQUID_prices,
            # "SQUID_vwap": self.SQUID_vwap
        })

        conversions = 1
        return result, conversions, traderData