from datamodel import OrderDepth, UserId, TradingState, Order, Observation, ConversionObservation
from typing import List, Dict, Tuple
import string
import jsonpickle
import numpy as np
import math
from math import log, sqrt, exp
from statistics import NormalDist


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
    

    def implied_bid_ask_macaroons(self, orderDep: OrderDepth, observation: ConversionObservation) -> Tuple[float, float]:
        """
        Calculate implied bid and ask prices for macaroons based on order depth and observation data.
        """
        buffer = 0.1    #! play around with this buffer to see if it improves performance
        return (
            observation.bidPrice - observation.transportFees - observation.exportTariff - buffer,
            observation.askPrice + observation.transportFees + observation.importTariff
        )
        
    def macaroons_take_orders(self, orderDep: OrderDepth, observation: ConversionObservation) -> Tuple[Order, Order]:
        """
        Take orders for macaroons based on implied bid/asks and observation data.
        """
        implied_bid, implied_ask = self.implied_bid_ask_macaroons(orderDep, observation)




    def run(self, state: TradingState):
        result = {}


        traderData = jsonpickle.encode({
        })

        conversions = 1
        return result, conversions, traderData