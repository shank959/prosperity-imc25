import json
import pandas as pd
from dataclasses import dataclass
from typing import List

@dataclass
class Trade:
    timestamp: int
    buyer: str
    seller: str
    symbol: str
    currency: str
    price: float
    quantity: int

# Read the log file
with open('./mylog.log', 'r') as f:
    content = f.read()

# Parse trades section
trades_start = content.find('Trade History:')
trades_section = content[trades_start + len('Trade History:'):].strip()
trades_data = json.loads(trades_section)

# Convert to Trade objects
trades: List[Trade] = [Trade(**t) for t in trades_data]

# Filter for our trades
our_trades = [trade for trade in trades if trade.buyer == "SUBMISSION" or trade.seller == "SUBMISSION"]

# Convert to DataFrame
trades_df = pd.DataFrame([trade.__dict__ for trade in our_trades])

# Calculate metrics for each symbol
symbols = ['PICNIC_BASKET1', 'CROISSANTS', 'JAMS', 'DJEMBES']
results = []

for symbol in symbols:
    symbol_trades = trades_df[trades_df['symbol'] == symbol]
    
    # Calculate buys
    buy_trades = symbol_trades[symbol_trades['buyer'] == 'SUBMISSION']
    total_buys = buy_trades['quantity'].sum()
    total_spent = (buy_trades['price'] * buy_trades['quantity']).sum()
    
    # Calculate sells
    sell_trades = symbol_trades[symbol_trades['seller'] == 'SUBMISSION']
    total_sells = sell_trades['quantity'].sum()
    total_made = (sell_trades['price'] * sell_trades['quantity']).sum()
    
    results.append({
        'Symbol': symbol,
        'Total Buys': total_buys,
        'Total Spent': round(total_spent, 2),
        'Total Sells': total_sells,
        'Total Made': round(total_made, 2),
        'Net Profit': round(total_made - total_spent, 2)
    })

# Convert results to DataFrame and display
results_df = pd.DataFrame(results)
print("\nTrade Analysis Results:")
print(results_df.to_string(index=False))

# Calculate overall totals
total_spent = results_df['Total Spent'].sum()
total_made = results_df['Total Made'].sum()
net_profit = total_made - total_spent

print("\nOverall Summary:")
print(f"Total Money Spent: {round(total_spent, 2)}")
print(f"Total Money Made: {round(total_made, 2)}")
print(f"Net Profit: {round(net_profit, 2)}") 