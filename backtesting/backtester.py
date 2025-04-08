# what i need for the backtester is
# something that evaluates the performance of the trading strategy
# based on historical data given by imc
# this historical data is a list of trades
# and list of top 3 bids and asks (price and volume) at eeach time point

class Backtester:
    def __init__(self, historical_data, trader_class, initial_capital=100000):
        self.historical_data = historical_data  # DataFrame with historical order books and trades
        self.trader_class = trader_class
        self.initial_capital = initial_capital
        self.positions = {}  # Current positions
        self.cash = initial_capital
        self.trades = []  # Record of all trades
        self.pnl_history = []  # P&L over time
        
    def run(self):
        # Sort data by timestamp
        self.historical_data = self.historical_data.sort_values('timestamp')
        
        # Initialize trader
        trader = self.trader_class()
        
        # For each timestamp
        for i, row in self.historical_data.iterrows():
            # Create TradingState from historical data
            state = self._create_trading_state(row)
            
            # Run trader strategy
            orders, conversions, trader_data = trader.run(state)
            
            # Simulate order execution
            self._execute_orders(orders, row)
            
            # Update P&L
            self._update_pnl(row)
            
        return self._calculate_performance_metrics()
    
    def _create_trading_state(self, row):
        # Convert row data to TradingState object
        # This would include creating OrderDepth objects from the top 3 bids/asks
        # ...
        
    def _execute_orders(self, orders, market_data):
        # Simulate order execution based on next timestamp's data
        # Update positions and cash
        # ...
        
    def _update_pnl(self, market_data):
        # Calculate P&L based on position changes and price movements
        # ...
        
    def _calculate_performance_metrics(self):
        # Calculate and return performance metrics
        # ...
