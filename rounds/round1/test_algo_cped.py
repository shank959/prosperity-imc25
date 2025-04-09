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
        self.KELP_prices = []
        self.KELP_vwap = []
        self.SQUID_prices = []
        self.SQUID_vwap = []
        self.SQUID_INK_prices = []
        self.SQUID_INK_volatility_history = []

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



    def rfr_orders(self, order_depth: OrderDepth, fair_value: int, width: int, position: int, position_limit: int) -> List[Order]:
        orders: List[Order] = []

        buy_order_volume = 0
        sell_order_volume = 0

        baaf = min(
            [price for price in order_depth.sell_orders.keys() if price > fair_value + 1])
        bbbf = max([price for price in order_depth.buy_orders.keys()
                   if price < fair_value - 1])
        #! TODO: TRY DIFFERENT HYPERPAREMETERS FOR MARKET MAKING (VOLOTAILITY CALCULATION)
        if len(order_depth.sell_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -1*order_depth.sell_orders[best_ask]
            if best_ask < fair_value:
                quantity = min(best_ask_amount, position_limit -
                               position)  # max amt to buy
                if quantity > 0:
                    orders.append(Order("RAINFOREST_RESIN", round(best_ask), quantity))
                    buy_order_volume += quantity

        if len(order_depth.buy_orders) != 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]
            if best_bid > fair_value:
                # should be the max we can sell
                quantity = min(best_bid_amount, position_limit + position)
                if quantity > 0:
                    orders.append(Order("RAINFOREST_RESIN", round(best_bid), -1 * quantity))
                    sell_order_volume += quantity

        buy_order_volume, sell_order_volume = self.clear_position_order(
            orders, order_depth, position, position_limit, "RAINFOREST_RESIN", buy_order_volume, sell_order_volume, fair_value, 1)

        buy_quantity = position_limit - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order("RAINFOREST_RESIN", round(bbbf + 1), buy_quantity))

        sell_quantity = position_limit + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order("RAINFOREST_RESIN", round(baaf - 1), -sell_quantity))

        return orders

    def clear_position_order(self, orders: List[Order], order_depth: OrderDepth, position: int, position_limit: int, product: str, buy_order_volume: int, sell_order_volume: int, fair_value: float, width: int) -> List[Order]:
        position_after_take = position + buy_order_volume - sell_order_volume
        fair_for_bid = math.floor(fair_value)
        fair_for_ask = math.ceil(fair_value)
        #! TODO: TRY OUT DIFFERENT METHODS OF OFLLOADING POSITION
        # fair_for_ask = fair_for_bid = fair

        buy_quantity = position_limit - (position + buy_order_volume)
        sell_quantity = position_limit + (position - sell_order_volume)

        if position_after_take > 0:
            if fair_for_ask in order_depth.buy_orders.keys():
                clear_quantity = min(
                    order_depth.buy_orders[fair_for_ask], position_after_take)
                #! TODO: SEE IF WE WANT TO OFFLOAD ENTIRE POSITIONS
                sent_quantity = min(sell_quantity, clear_quantity)
                orders.append(Order(product, round(fair_for_ask), -abs(sent_quantity)))
                sell_order_volume += abs(sent_quantity)

        if position_after_take < 0:
            if fair_for_bid in order_depth.sell_orders.keys():
                clear_quantity = min(
                    abs(order_depth.sell_orders[fair_for_bid]), abs(position_after_take))
                # clear_quantity = abs(position_after_take)
                sent_quantity = min(buy_quantity, clear_quantity)
                orders.append(Order(product, round(fair_for_bid), abs(sent_quantity)))
                buy_order_volume += abs(sent_quantity)

        return buy_order_volume, sell_order_volume

    # Method: mid_price,
    def KELP_fair_value(self, order_depth: OrderDepth, method="mid_price", min_vol=0) -> float:
        if method == "mid_price":
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            mid_price = (best_ask + best_bid) / 2
            return mid_price
        #! TODO: TEST OUT DIFFERENT METHODS OF CALCULATING FAIR VALUE MM VWAP ECT

        elif method == "mid_price_with_vol_filter":
            if len([price for price in order_depth.sell_orders.keys() if abs(order_depth.sell_orders[price]) >= min_vol]) == 0 or len([price for price in order_depth.buy_orders.keys() if abs(order_depth.buy_orders[price]) >= min_vol]) == 0:
                best_ask = min(order_depth.sell_orders.keys())
                best_bid = max(order_depth.buy_orders.keys())
                return (best_ask + best_bid) / 2
            else:
                best_ask = min([price for price in order_depth.sell_orders.keys() if abs(order_depth.sell_orders[price]) >= min_vol])
                best_bid = max([price for price in order_depth.buy_orders.keys() if abs(order_depth.buy_orders[price]) >= min_vol])
                return (best_ask + best_bid) / 2

    def KELP_orders(self, order_depth: OrderDepth, timespan: int, width: float, KELP_take_width: float, position: int, position_limit: int) -> List[Order]:
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
            self.KELP_prices.append(mmmid_price)

            volume = -1 * \
                order_depth.sell_orders[best_ask] + \
                order_depth.buy_orders[best_bid]
            vwap = (best_bid * (-1) * order_depth.sell_orders[best_ask] +
                    best_ask * order_depth.buy_orders[best_bid]) / volume
            self.KELP_vwap.append({"vol": volume, "vwap": vwap})

            if len(self.KELP_vwap) > timespan:
                self.KELP_vwap.pop(0)

            if len(self.KELP_prices) > timespan:
                self.KELP_prices.pop(0)

            fair_value = mmmid_price

            if best_ask <= fair_value - KELP_take_width:
                ask_amount = -1 * order_depth.sell_orders[best_ask]
                if ask_amount <= 20:    #! WHY 20? WHY NOT 15 LIKE EARLIER? ARE WE TO ASSUME 20 IS OPTIMAL THRESDHOLD FOR POS EV? 
                    quantity = min(ask_amount, position_limit - position)
                    if quantity > 0:
                        orders.append(Order("KELP", round(best_ask), quantity))
                        buy_order_volume += quantity

            if best_bid >= fair_value + KELP_take_width:
                bid_amount = order_depth.buy_orders[best_bid]
                if bid_amount <= 20:
                    quantity = min(bid_amount, position_limit + position)
                    if quantity > 0:
                        orders.append(Order("KELP", round(best_bid), -1 * quantity))
                        sell_order_volume += quantity

            buy_order_volume, sell_order_volume = self.clear_position_order(
                orders, order_depth, position, position_limit, "KELP", buy_order_volume, sell_order_volume, fair_value, 2)

            #! MARKET MAKING, TRY NEW METHODS with different thresholders

            aaf = [price for price in order_depth.sell_orders.keys() if price > fair_value + 1]
            bbf = [price for price in order_depth.buy_orders.keys() if price < fair_value - 1]
            baaf = min(aaf) if aaf else fair_value + 2
            bbbf = max(bbf) if bbf else fair_value - 2

            buy_quantity = position_limit - (position + buy_order_volume)
            if buy_quantity > 0:
                orders.append(Order("KELP", round(bbbf + 1), buy_quantity))

            sell_quantity = position_limit + (position - sell_order_volume)
            if sell_quantity > 0:
                orders.append(Order("KELP", round(baaf - 1), -sell_quantity))

        return orders

    def SQUID_orders(self, order_depth: OrderDepth, timespan: int, make_width: float, take_width: float, position: int, position_limit: int) -> List[Order]:
        orders: List[Order] = []

        if not order_depth.sell_orders or not order_depth.buy_orders:
            return orders

        best_ask = min(order_depth.sell_orders.keys())
        best_bid = max(order_depth.buy_orders.keys())
        mid_price = (best_ask + best_bid) / 2.0

        offset = 1.0  # ! TODO: adjust

        if position < position_limit:
            buy_qty = position_limit - position
            orders.append(Order("SQUID_INK", round(mid_price - offset), buy_qty))

        if position > -position_limit:
            sell_qty = position_limit + position
            orders.append(Order("SQUID_INK", round(mid_price + offset), -sell_qty))

        return orders

    def squid_ink_fetch_historical_data(self):
        day_price_0_df = pd.read_csv(
            "../round-1-island-data-bottle/prices_round_1_day_0.csv", delimiter=";")
        day_price_m1_df = pd.read_csv(
            "../round-1-island-data-bottle/prices_round_1_day_-1.csv", delimiter=";")
        day_price_m2_df = pd.read_csv(
            "../round-1-island-data-bottle/prices_round_1_day_-2.csv", delimiter=";")

        merged_df = pd.concat(
            [day_price_0_df, day_price_m1_df, day_price_m2_df])

        squid_ink_df = merged_df[merged_df["product"] == "SQUID_INK"].copy()
        return squid_ink_df

    def squid_ink_mean_reversion(self):
        # We want to combine this strategy with others to do directional trading at specific points
        z_score_threshold = 1.5  # TODO: tune hyperparameter

        squid_ink_df = self.squid_ink_fetch_historical_data()

        mean_price = squid_ink_df['mid_price'].mean()
        std_price = squid_ink_df['mid_price'].std()

        if std_price == 0:
            squid_ink_df['z_score'] = 0
        else:
            squid_ink_df['z_score'] = (
                squid_ink_df['mid_price'] - mean_price) / std_price

        squid_ink_df['mr_signal'] = 0  # 0 = hold, 1 = buy, -1 = sell
        squid_ink_df.loc[squid_ink_df['z_score'] < -
                         z_score_threshold, 'mr_signal'] = 1  # Buy signal
        squid_ink_df.loc[squid_ink_df['z_score'] >
                         z_score_threshold, 'mr_signal'] = -1  # Sell signal

        return squid_ink_df

    def momentum_strategy(self, lookback=1):
        lookback = 3  # TODO: tune hyperparameter
        squid_ink_df = self.squid_ink_fetch_historical_data()
        squid_ink_df['price_change'] = squid_ink_df['mid_price'].diff(lookback)
        squid_ink_df['momentum_signal'] = 0
        squid_ink_df.loc[squid_ink_df['price_change']
                         > 0, 'momentum_signal'] = 1  # Buy signal
        squid_ink_df.loc[squid_ink_df['price_change']
                         < 0, 'momentum_signal'] = -1  # Sell signal

        return squid_ink_df

    # TODO: might want to add more strategies before combining

    # TODO: integrate buy/sell signal logic with SQUID_orders so we can add the orders to the order book

    def run(self, state: TradingState):
        result = {}

        rfr_fair_value = 10000  # Participant should calculate this value
        rfr_width = 2
        rfr_position_limit = 50

        # ! TODO: CHANGE TO WIDTH BEING A FUNCTION OF DIFFERENCE WITH FAIR VALUE AND VOLATILITY
        KELP_make_width = 3.5
        # ! TODO: CHANGE TO WIDTH BEING A FUNCTION OF DIFFERENCE WITH FAIR VALUE AND VOLATILITY
        KELP_take_width = 1
        KELP_position_limit = 50
        KELP_timemspan = 10

        squid_make_width = 3.0  # ! TODO: adjust
        squid_take_width = 1.0  # ! TODO: adjust
        squid_position_limit = 50
        squid_timespan = 10

        # traderData = jsonpickle.decode(state.traderData)
        # print(state.traderData)
        # self.KELP_prices = traderData["KELP_prices"]
        # self.KELP_vwap = traderData["KELP_vwap"]

        if "RAINFOREST_RESIN" in state.order_depths:
            rfr_position = state.position["RAINFOREST_RESIN"] if "RAINFOREST_RESIN" in state.position else 0
            rfr_orders = self.rfr_orders(
                state.order_depths["RAINFOREST_RESIN"], rfr_fair_value, rfr_width, rfr_position, rfr_position_limit)
            result["RAINFOREST_RESIN"] = rfr_orders

        if "KELP" in state.order_depths:
            KELP_position = state.position["KELP"] if "KELP" in state.position else 0
            KELP_orders = self.KELP_orders(
                state.order_depths["KELP"], KELP_timemspan, KELP_make_width, KELP_take_width, KELP_position, KELP_position_limit)
            result["KELP"] = KELP_orders

        if "SQUID_INK" in state.order_depths:
            squid_position = state.position["SQUID_INK"] if "SQUID_INK" in state.position else 0
            squid_orders = self.SQUID_orders(
                state.order_depths["SQUID_INK"], squid_timespan, squid_make_width, squid_take_width, squid_position, squid_position_limit)
            result["SQUID_INK"] = squid_orders

        traderData = jsonpickle.encode({
            "KELP_prices": self.KELP_prices,
            "KELP_vwap": self.KELP_vwap,
            "SQUID_prices": self.SQUID_prices,
            "SQUID_vwap": self.SQUID_vwap
        })

        conversions = 1
        return result, conversions, traderData
