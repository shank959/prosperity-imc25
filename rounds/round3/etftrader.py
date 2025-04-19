# spread trading and tryna rb between basket and synt
from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List

class Trader:
    #hardcoded const
    PREMIUM   = {"PICNIC_BASKET1": 50,  "PICNIC_BASKET2": 40}  # theta
    POS_LIMIT = {"PICNIC_BASKET1": 60,  "PICNIC_BASKET2": 100}

    # b1: symmetric/aggressive
    TRIG_UP_B1 =  85 #(old: +/- 75)
    TRIG_DN_B1 = -85
    STEP_B1 = 60 # go straight to limit

    # b2: assym and safer
    TRIG_UP_B2 = 90 #sell

    TRIG_DN_B2 = -105 # buy (used to be -90)
    STOP_LOSS_B2 = -160 # new testing stoploss (old: -140)

    STEP_B2 = 20 #(used to be 20)
    EXIT_BAND = 10 # flatten when |spread–theta| < 20 (old: 20)

    RECIPE = {
        "PICNIC_BASKET1": {"CROISSANTS": 6, "JAMS": 3, "DJEMBES": 1},
        "PICNIC_BASKET2": {"CROISSANTS": 4, "JAMS": 2}
    }

    @staticmethod
    def best_bid(depth: OrderDepth) -> int:
        return max(depth.buy_orders.keys())

    @staticmethod
    def best_ask(depth: OrderDepth) -> int:
        return min(depth.sell_orders.keys())

    def mid(self, depth: OrderDepth) -> float:
        return (self.best_bid(depth) + self.best_ask(depth)) / 2

    def synthetic_mid(self, basket: str, depths: Dict[str, OrderDepth]) -> float:
        return sum(self.mid(depths[p]) * qty for p, qty in self.RECIPE[basket].items())


    def trade_basket1(self, depths: Dict[str, OrderDepth], pos: int) -> List[Order]:
        depth    = depths["PICNIC_BASKET1"]
        spread   = self.mid(depth) - self.synthetic_mid("PICNIC_BASKET1", depths) - self.PREMIUM["PICNIC_BASKET1"]
        orders   : List[Order] = []
        limit    = self.POS_LIMIT["PICNIC_BASKET1"]

        # SELL if overpriced
        if spread > self.TRIG_UP_B1 and pos > -limit:
            qty = min(self.STEP_B1, limit + pos,
                      depth.buy_orders[self.best_bid(depth)])
            if qty > 0:
                orders.append(Order("PICNIC_BASKET1", self.best_bid(depth), -qty))

        # BUY if under
        if spread < self.TRIG_DN_B1 and pos <  limit:
            qty = min(self.STEP_B1, limit - pos,
                      abs(depth.sell_orders[self.best_ask(depth)]))
            if qty > 0:
                orders.append(Order("PICNIC_BASKET1", self.best_ask(depth),  qty))

        return orders

    def trade_basket2(self, depths: Dict[str, OrderDepth], pos: int) -> List[Order]:
        depth    = depths["PICNIC_BASKET2"]
        theta    = self.PREMIUM["PICNIC_BASKET2"]
        spread   = self.mid(depth) - self.synthetic_mid("PICNIC_BASKET2", depths) - theta
        orders   : List[Order] = []
        limit    = self.POS_LIMIT["PICNIC_BASKET2"]

        # exit/flatten if spread is returning to thresh we set
        if abs(spread) < self.EXIT_BAND and pos != 0:
            if pos > 0:
                qty = min(pos, depth.buy_orders[self.best_bid(depth)])
                orders.append(Order("PICNIC_BASKET2", self.best_bid(depth), -qty))
            else:
                qty = min(-pos, abs(depth.sell_orders[self.best_ask(depth)]))
                orders.append(Order("PICNIC_BASKET2", self.best_ask(depth),  qty))
            return orders

        # SELL (over priced)
        if spread > self.TRIG_UP_B2 and pos > -limit:
            qty = min(self.STEP_B2, limit + pos,
                      depth.buy_orders[self.best_bid(depth)])
            if qty > 0:
                orders.append(Order("PICNIC_BASKET2", self.best_bid(depth), -qty))

        # BUY (under priced)
        if spread < self.TRIG_DN_B2 - 20 and pos < limit:
            qty = min(self.STEP_B2, limit - pos,
                      abs(depth.sell_orders[self.best_ask(depth)]))
            if qty > 0:
                orders.append(Order("PICNIC_BASKET2", self.best_ask(depth),  qty))
        
        # hard stoploss (!TODO)
        if spread < self.STOP_LOSS_B2 and pos > 0:
            qty = min(pos, depth.buy_orders[self.best_bid(depth)])
            orders.append(Order("PICNIC_BASKET2", self.best_bid(depth), -qty))
            return orders

        return orders

    def run(self, state: TradingState):
        depths  = state.order_depths
        result: Dict[str, List[Order]] = {}

        pos1 = state.position.get("PICNIC_BASKET1", 0)
        result["PICNIC_BASKET1"] = self.trade_basket1(depths, pos1)

        if "PICNIC_BASKET2" in depths:
            pos2 = state.position.get("PICNIC_BASKET2", 0)
            result["PICNIC_BASKET2"] = self.trade_basket2(depths, pos2)

        return result, 0, ""