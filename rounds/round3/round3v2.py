from datamodel import OrderDepth, TradingState, Order
import numpy as np
import jsonpickle
from typing import Dict, List, Any

class Trader:
    def __init__(self):
        # Adjusted parameters to reflect an average $50 spread and separate thresholds.
        self.params = {
            "SPREAD": {
                "spread_std_window": 45,       # Number of ticks for std dev calculation.
                "spread_sma_window": 150,        # Number of ticks for rolling mean.
                "default_spread_mean": 50,       # Default mean spread (expected average difference).
                "buy_zscore_threshold": 0.5,     # Lower threshold to trigger basket BUY (and underlying SELL).
                "sell_zscore_threshold": 2.0,    # Higher threshold to trigger basket SELL (and underlying BUY).
                "target_position_buy": 40,       # When buying baskets, aim for a positive target of +40.
                "target_position_sell": 50       # When selling baskets, aim for a negative target of -50.
            }
        }
    
    def get_swmid(self, order_depth: OrderDepth) -> float:
        """Compute mid-price as average of best bid and best ask."""
        if order_depth.buy_orders and order_depth.sell_orders:
            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys())
            return (best_bid + best_ask) / 2.0
        elif order_depth.buy_orders:
            return max(order_depth.buy_orders.keys())
        elif order_depth.sell_orders:
            return min(order_depth.sell_orders.keys())
        else:
            return 0

    def get_picnic_basket1_order_depth(self, order_depths: Dict[str, OrderDepth]) -> OrderDepth:
        """Build synthetic basket depth from underlying components."""
        CROISSANTS_PER_BASKET = 6
        JAMS_PER_BASKET = 3
        DJEMBES_PER_BASKET = 1

        synthetic_order_price = OrderDepth()

        # Underlying best bid/ask.
        croissants_bid = max(order_depths["CROISSANTS"].buy_orders.keys(), default=0)
        croissants_ask = min(order_depths["CROISSANTS"].sell_orders.keys(), default=float('inf'))
        jams_bid = max(order_depths["JAMS"].buy_orders.keys(), default=0)
        jams_ask = min(order_depths["JAMS"].sell_orders.keys(), default=float('inf'))
        djembes_bid = max(order_depths["DJEMBES"].buy_orders.keys(), default=0)
        djembes_ask = min(order_depths["DJEMBES"].sell_orders.keys(), default=float('inf'))

        # Synthetic basket prices.
        implied_bid = (croissants_bid * CROISSANTS_PER_BASKET +
                       jams_bid * JAMS_PER_BASKET +
                       djembes_bid * DJEMBES_PER_BASKET)
        implied_ask = (croissants_ask * CROISSANTS_PER_BASKET +
                       jams_ask * JAMS_PER_BASKET +
                       djembes_ask * DJEMBES_PER_BASKET)

        # Available volume (expressed in number of baskets).
        if implied_bid > 0:
            croissants_bid_vol = order_depths["CROISSANTS"].buy_orders.get(croissants_bid, 0) // CROISSANTS_PER_BASKET
            jams_bid_vol = order_depths["JAMS"].buy_orders.get(jams_bid, 0) // JAMS_PER_BASKET
            djembes_bid_vol = order_depths["DJEMBES"].buy_orders.get(djembes_bid, 0) // DJEMBES_PER_BASKET
            synthetic_bid_vol = min(croissants_bid_vol, jams_bid_vol, djembes_bid_vol)
            synthetic_order_price.buy_orders[implied_bid] = synthetic_bid_vol

        if implied_ask < float('inf'):
            croissants_ask_vol = -order_depths["CROISSANTS"].sell_orders.get(croissants_ask, 0) // CROISSANTS_PER_BASKET
            jams_ask_vol = -order_depths["JAMS"].sell_orders.get(jams_ask, 0) // JAMS_PER_BASKET
            djembes_ask_vol = -order_depths["DJEMBES"].sell_orders.get(djembes_ask, 0) // DJEMBES_PER_BASKET
            synthetic_ask_vol = min(croissants_ask_vol, jams_ask_vol, djembes_ask_vol)
            synthetic_order_price.sell_orders[implied_ask] = -synthetic_ask_vol

        return synthetic_order_price

    def convert_picnic_basket1_orders(self, synthetic_orders: List[Order], order_depths: Dict[str, OrderDepth]) -> Dict[str, List[Order]]:
        """Convert synthetic orders into underlying orders in fixed ratio."""
        component_orders = {"CROISSANTS": [], "JAMS": [], "DJEMBES": []}
        synthetic_basket_order_depth = self.get_picnic_basket1_order_depth(order_depths)
        best_bid = max(synthetic_basket_order_depth.buy_orders.keys()) if synthetic_basket_order_depth.buy_orders else 0
        best_ask = min(synthetic_basket_order_depth.sell_orders.keys()) if synthetic_basket_order_depth.sell_orders else float("inf")
        CROISSANTS_PER_BASKET = 6
        JAMS_PER_BASKET = 3
        DJEMBES_PER_BASKET = 1

        for order in synthetic_orders:
            price = order.price
            quantity = order.quantity  # Positive means BUY, negative means SELL
            if quantity > 0 and price >= best_ask:
                # For a BUY basket order, we hedge by buying underlyings at best ask prices.
                croissants_price = min(order_depths["CROISSANTS"].sell_orders.keys())
                jams_price = min(order_depths["JAMS"].sell_orders.keys())
                djembes_price = min(order_depths["DJEMBES"].sell_orders.keys())
            elif quantity < 0 and price <= best_bid:
                # For a SELL basket order, hedge by selling underlyings at best bid prices.
                croissants_price = max(order_depths["CROISSANTS"].buy_orders.keys())
                jams_price = max(order_depths["JAMS"].buy_orders.keys())
                djembes_price = max(order_depths["DJEMBES"].buy_orders.keys())
            else:
                continue

            croissants_order = Order("CROISSANTS", croissants_price, quantity * CROISSANTS_PER_BASKET)
            jams_order = Order("JAMS", jams_price, quantity * JAMS_PER_BASKET)
            djembes_order = Order("DJEMBES", djembes_price, quantity * DJEMBES_PER_BASKET)
            component_orders["CROISSANTS"].append(croissants_order)
            component_orders["JAMS"].append(jams_order)
            component_orders["DJEMBES"].append(djembes_order)

        return component_orders

    def available_baskets_for_selling_underlyings(self, order_depths: Dict[str, OrderDepth]) -> float:
        """Return maximum baskets that can be hedged by selling underlyings (from buy orders)."""
        available = []
        for underlying, weight in [("CROISSANTS", 6), ("JAMS", 3), ("DJEMBES", 1)]:
            if order_depths[underlying].buy_orders:
                best_bid = max(order_depths[underlying].buy_orders.keys())
                volume = order_depths[underlying].buy_orders[best_bid]
                available.append(volume // weight)
            else:
                available.append(0)
        return min(available)

    def available_baskets_for_buying_underlyings(self, order_depths: Dict[str, OrderDepth]) -> float:
        """Return maximum baskets that can be hedged by buying underlyings (from sell orders)."""
        available = []
        for underlying, weight in [("CROISSANTS", 6), ("JAMS", 3), ("DJEMBES", 1)]:
            if order_depths[underlying].sell_orders:
                best_ask = min(order_depths[underlying].sell_orders.keys())
                volume = abs(order_depths[underlying].sell_orders[best_ask])
                available.append(volume // weight)
            else:
                available.append(0)
        return min(available)

    def execute_picnic_basket1_spread_orders(self, target_position: int, basket_position: int, order_depths: Dict[str, OrderDepth]):
        """
        Execute the basket spread order.
          - If target_position > basket_position: we want to BUY baskets (hedge by SELLING underlyings).
          - If target_position < basket_position: we want to SELL baskets (hedge by BUYING underlyings).
        """
        if target_position == basket_position:
            return None

        target_quantity = abs(target_position - basket_position)
        basket_order_depth = order_depths["PICNIC_BASKET1"]
        synthetic_order_depth = self.get_picnic_basket1_order_depth(order_depths)

        if target_position > basket_position:
            # Trade Type: BUY basket; hedge by SELLING underlyings.
            basket_ask_price = min(basket_order_depth.sell_orders.keys())
            basket_ask_volume = abs(basket_order_depth.sell_orders[basket_ask_price])
            synthetic_bid_price = max(synthetic_order_depth.buy_orders.keys())
            synthetic_bid_volume = abs(synthetic_order_depth.buy_orders[synthetic_bid_price])
            orderbook_volume = min(basket_ask_volume, synthetic_bid_volume)
            execute_volume = min(orderbook_volume, target_quantity)
            # Limit execution to what underlyings can hedge.
            underlying_max = self.available_baskets_for_selling_underlyings(order_depths)
            execute_volume = min(execute_volume, underlying_max)
            basket_orders = [Order("PICNIC_BASKET1", basket_ask_price, execute_volume)]
            # When buying the basket, hedge by SELLING underlyings → synthetic order is negative.
            synthetic_orders = [Order("PICNIC_SYNTHETIC", synthetic_bid_price, -execute_volume)]
            aggregate_orders = self.convert_picnic_basket1_orders(synthetic_orders, order_depths)
            aggregate_orders["PICNIC_BASKET1"] = basket_orders
            return aggregate_orders
        else:
            # Trade Type: SELL basket; hedge by BUYING underlyings.
            basket_bid_price = max(basket_order_depth.buy_orders.keys())
            basket_bid_volume = abs(basket_order_depth.buy_orders[basket_bid_price])
            synthetic_ask_price = min(synthetic_order_depth.sell_orders.keys())
            synthetic_ask_volume = abs(synthetic_order_depth.sell_orders[synthetic_ask_price])
            orderbook_volume = min(basket_bid_volume, synthetic_ask_volume)
            execute_volume = min(orderbook_volume, target_quantity)
            # Limit execution by available liquidity on underlying sell orders.
            underlying_max = self.available_baskets_for_buying_underlyings(order_depths)
            execute_volume = min(execute_volume, underlying_max)
            basket_orders = [Order("PICNIC_BASKET1", basket_bid_price, -execute_volume)]
            # When selling the basket, hedge by BUYING underlyings → synthetic order is positive.
            synthetic_orders = [Order("PICNIC_SYNTHETIC", synthetic_ask_price, execute_volume)]
            aggregate_orders = self.convert_picnic_basket1_orders(synthetic_orders, order_depths)
            aggregate_orders["PICNIC_BASKET1"] = basket_orders
            return aggregate_orders

    def spread_picnic_basket1_orders(self, order_depths: Dict[str, OrderDepth], product: str, basket_position: int, spread_data: Dict[str, Any]):
        if "PICNIC_BASKET1" not in order_depths.keys():
            return None

        basket_order_depth = order_depths["PICNIC_BASKET1"]
        synthetic_order_depth = self.get_picnic_basket1_order_depth(order_depths)
        basket_swmid = self.get_swmid(basket_order_depth)
        synthetic_swmid = self.get_swmid(synthetic_order_depth)
        spread = basket_swmid - synthetic_swmid
        spread_data["spread_history"].append(spread)

        if len(spread_data["spread_history"]) < self.params["SPREAD"]["spread_std_window"]:
            return None
        else:
            spread_std = np.std(spread_data["spread_history"][-self.params["SPREAD"]["spread_std_window"]:])
        
        if len(spread_data["spread_history"]) == self.params["SPREAD"]["spread_sma_window"]:
            spread_mean = np.mean(spread_data["spread_history"])
            spread_data["curr_mean"] = spread_mean
        elif len(spread_data["spread_history"]) > self.params["SPREAD"]["spread_sma_window"]:
            spread_mean = spread_data["curr_mean"] + ((spread - spread_data["spread_history"][0]) / self.params["SPREAD"]["spread_sma_window"])
            spread_data["spread_history"].pop(0)
        else:
            spread_mean = self.params["SPREAD"]["default_spread_mean"]

        zscore = (spread - spread_mean) / (spread_std if spread_std != 0 else 1.0)

        # Use separate thresholds:
        if zscore >= self.params["SPREAD"]["sell_zscore_threshold"]:
            # Basket is too expensive → sell basket (i.e., target negative basket position)
            if basket_position != -self.params["SPREAD"]["target_position_sell"]:
                return self.execute_picnic_basket1_spread_orders(-self.params["SPREAD"]["target_position_sell"], basket_position, order_depths)

        if zscore <= -self.params["SPREAD"]["buy_zscore_threshold"]:
            # Basket is cheap → buy basket (i.e., target positive basket position)
            if basket_position != self.params["SPREAD"]["target_position_buy"]:
                return self.execute_picnic_basket1_spread_orders(self.params["SPREAD"]["target_position_buy"], basket_position, order_depths)

        spread_data["prev_zscore"] = zscore
        return None

    def run(self, state: TradingState):
        result = {}
        trader_object = {}
        if state.traderData:
            trader_object = jsonpickle.decode(state.traderData)
        if "SPREAD" not in trader_object:
            trader_object["SPREAD"] = {"spread_history": [], "curr_mean": 0, "prev_zscore": 0}

        # Set new target parameters for buy and sell
        # (These values can be tweaked based on performance.)
        self.params["SPREAD"]["buy_zscore_threshold"] = 0.5
        self.params["SPREAD"]["sell_zscore_threshold"] = 2.0
        self.params["SPREAD"]["target_position_buy"] = 40
        self.params["SPREAD"]["target_position_sell"] = 50

        basket_position = state.position["PICNIC_BASKET1"] if "PICNIC_BASKET1" in state.position else 0
        spread_orders = self.spread_picnic_basket1_orders(state.order_depths, "PICNIC_BASKET1", basket_position, trader_object["SPREAD"])
        if spread_orders is not None:
            result["CROISSANTS"] = spread_orders["CROISSANTS"]
            result["JAMS"] = spread_orders["JAMS"]
            result["DJEMBES"] = spread_orders["DJEMBES"]
            result["PICNIC_BASKET1"] = spread_orders["PICNIC_BASKET1"]

        traderData = jsonpickle.encode(trader_object)
        conversions = 0
        return result, conversions, traderData
