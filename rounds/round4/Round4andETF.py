from datamodel import OrderDepth, UserId, TradingState, Order, Observation, ConversionObservation
from typing import List, Dict, Tuple, Any
import string
import jsonpickle
import numpy as np
import math
from math import log, sqrt, exp
from statistics import NormalDist


class Trader:
    def __init__(self):
        self.MACARONS = {
            "position_limit": 75,
            "conversion_limit": 10,
            "storage_cost": 0.1,
            "critical_sunlight_index": 45,
            "sunlight_data": [],
            "mean_price": 630,
            "sell_band_low": 670,
            "buy_band_high": 590,
            "in_panic_mode": False
        }
        # ------------------------ BASKET TRADING ------------------------
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
        # ---------------------------------------------------------------

    def macaron_get_sunlight_state(self, observation: ConversionObservation) -> Dict[str, Any]:
        self.MACARONS["sunlight_data"].append(observation.sunlightIndex)
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
            layer_qty = min(avail, remaining * 0.5)
            orders.append(Order("MAGNIFICENT_MACARONS", price, layer_qty))
            remaining -= layer_qty

        # 6) Take‑profit sells around the *local* mid
        local_mid = (best_ask + best_bid) // 2
        # profit_spread can be more than 1 if local bid/ask spreads are wide
        profit_spread =  max(2, (best_ask - best_bid) // 2)
        exit_price   = local_mid + profit_spread
        exit_qty     = position * 0.2
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
        alpha = delta / roc # timestamps until above critical
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
        low_s, high_s = m["sell_band_low"], m["sell_band_high"]
        low_b, high_b = m["buy_band_low"],  m["buy_band_high"]
        cap_limit     = m["position_limit"]

        orders = []
        buy_volume = 0
        sell_volume = 0
        depth = orderDep
        # SELL band
        best_bid = max(depth.buy_orders.keys())
        best_bid_volume = depth.buy_orders[best_bid]
        if best_bid > low_s:
            orders.append(Order("MAGNIFICENT_MACARONS", best_bid, -best_bid_volume))
            sell_volume += best_bid_volume
        # BUY band
        best_ask = min(depth.sell_orders.keys())
        best_ask_volume = depth.sell_orders[best_ask]
        if best_ask < high_b:
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

    # ---------------------------------------------------------------

    def run(self, state: TradingState):
        result = {}
        conversions = 0
        buy_order_volume = 0
        sell_order_volume = 0

        traderObject = {"position": 0}
        if state.traderData:
            traderObject = jsonpickle.decode(state.traderData)

        try:
            if state.own_trades.get("MAGNIFICENT_MACARONS", 0):
                print(f"own_trades: {state.own_trades.get('MAGNIFICENT_MACARONS', 0)}")
        except:
            pass


        if "MAGNIFICENT_MACARONS" in state.order_depths:
            # check if in panic mode
            observation = state.observations.conversionObservations["MAGNIFICENT_MACARONS"]
            sunlight_state = self.macaron_get_sunlight_state(observation)

            # print all data
            print(f"sunlight_index: {state.observations.conversionObservations['MAGNIFICENT_MACARONS'].sunlightIndex}")
            print(f"best_bid: {sorted(state.order_depths['MAGNIFICENT_MACARONS'].buy_orders.keys(), reverse=True)}")
            print(f"best_ask: {sorted(state.order_depths['MAGNIFICENT_MACARONS'].sell_orders.keys())}")
            print(f"midprice: {(sorted(state.order_depths['MAGNIFICENT_MACARONS'].buy_orders.keys(), reverse=True)[0] + sorted(state.order_depths['MAGNIFICENT_MACARONS'].sell_orders.keys())[0]) / 2}")
            implied_bid, implied_ask = self.implied_bid_ask_macarons(observation)
            print(f"implied_bid: {implied_bid}")
            print(f"implied_ask: {implied_ask}")


            if sunlight_state["is_panic_mode"] and not self.MACARONS["in_panic_mode"]:
                print(f"panic mode: {sunlight_state['sunlight_index']}")
                orders = self.panic_mode_orders(state.order_depths["MAGNIFICENT_MACARONS"], observation, traderObject["position"])
            else:
                # orders, buy_order_volume, sell_order_volume = self.macaron_band_orders(state.order_depths["MAGNIFICENT_MACARONS"], traderObject["position"])
                orders = []
            # else:
            #     orders, buy_order_volume, sell_order_volume = self.macarons_take_orders(state.order_depths["MAGNIFICENT_MACARONS"], observation)
            #     conversions = self.macarons_conversions(buy_order_volume - sell_order_volume)
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
            depths = state.order_depths
            
            if "PICNIC_BASKET1" in depths:
                pos1 = state.position.get("PICNIC_BASKET1", 0)
                result["PICNIC_BASKET1"] = self.trade_basket1(depths, pos1)

            if "PICNIC_BASKET2" in depths:
                pos2 = state.position.get("PICNIC_BASKET2", 0)
                result["PICNIC_BASKET2"] = self.trade_basket2(depths, pos2)
            # ---------------------------------------------------------------
  
            # print(f"conversions: {conversions}")
            # print(f"buy_order_volume: {buy_order_volume}")
            # print(f"sell_order_volume: {sell_order_volume}")
            print(f"position: {traderObject['position']}")
            print(f"orders: {orders}")

        traderData = jsonpickle.encode({
            "position": traderObject["position"] + (buy_order_volume - sell_order_volume) + conversions,
            "in_panic_mode": sunlight_state["is_panic_mode"]
        })

        print(result)


        return result, conversions, traderData