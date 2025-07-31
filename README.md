# Prosperity IMC Trading Challenge 2025

This repository contains the algorithms, exploratory data analysis (EDA), and results for my participation in the IMC Prosperity Algorithmic Trading Challenge. Each round of the competition has its own dedicated folder containing the final code submission and related analysis.

---

## Repository Structure

The repository is organized by competition rounds. Each round's materials are located in a corresponding directory within the `rounds/` folder.

A typical directory for a round (e.g., `rounds/round1/`) contains:

* **`RoundX.py`**: The final Python file with the `Trader` class and the trading logic submitted for that round.

* Since the strategies used in each of the rounds are reused in subsequent rounds, one can refer to `rounds/round5/Round5.py` for implementation of each item.

---

## Round 1

### Strategy

Our approach combined market making and market taking strategies:

**Core Components:**
- **Market Making**: Provide liquidity at advantageous prices
- **Market Taking**: Lift undervalued offers and overvalued bids
- **Fair Value Calculation**: Essential for determining optimal entry/exit points
- **Inventory Management**: Systematic position clearing due to position limits

**Fair Value Calculations by Asset:**
- **Rainforest Resin**: Fixed fair value based on historical price data
- **Kelp**: Weighted average of large bids (assumed to be market maker orders)

- **Squid Ink**: Enhanced Kelp strat with mean reversion beta parameter. 

**Results & Improvements:**
- Achieved strong performance in Round 1
- Potential improvements: More dynamic mean reversion estimator 


---

## Round 2

### Strategy

Mainly did market taking strategies with **ETF arbitrage** on the plain ETF and components and also on synthetic instruments:

**Core Components:**
- **ETF Arbitrage**: Trade baskets against their underlying components
- **Synthetic Instrument Creation**: Built synthetic DJEMBES for arbitrage opportunities
- **Multi-Product Position Management**: This was hard because each of the strategies involved each of the products so at every point, there needed to be an inventory check and also a current order check.

**Asset-Specific Strategies:**

**Picnic Basket Arbitrade:**
- **Fair Value Calculation**: The baskets often traded at a premium. We simply used an average of the historical premiums.

- **ETF Arbitrage**: We would by the compents and sell the basket or vice versa and close the position when there was convergence.

**Synthetic DJEMBES Arbitrage:**
- **Synthetic Construction**: Created synthetic DJEMBES using 2×BASKET1 - 3×BASKET2 relationship then did the same ETF arb strat.



**Results & Improvements:**
- This gave moderate returns but we did not conduct market making for these items because the bid ask spread was really tight. 
- Potential improvements: 
    - The calculation of the premium felt a bit arbitrary and we were not too sure how to estimate it. In practice we just used an average of the historical premium. 
    - There were definitely better ways we couldve closed our positions for eg. by tracking pnl of each term structure.

---

## Round 3

### Strategy

Implemented **options trading with Black-Scholes pricing** and implied volatility calculations:

**Core Components:**
- **Implied Volatility Calculation**: Used numerical methods to extract implied volatility (IV)
- **Volatility Surface Fitting**: Polynomial fitting as suggested by the hint across strike prices for fair value estimation
- **Delta Hedging**: Position management using calculated option deltas
- **Market Making/Taking**: Applied to options with computed fair values

**Technical Implementation:**

**Implied Volatility Methods:**
We started with a bisection method for calculating implied volatility which was stable and gave robust convergence. I also tried implementing Newton-Raphson which converged much faster, but it caused a lot of instability in practice so we ended up sticking with the bisection method since reliability was more important than speed.

**Fair Value Calculation:**
For the volatility surface, we used a polynomial fit (degree 2) across moneyness as suggested by the hint. Then we plugged the fitted IV back into Black-Scholes to get our theoretical option prices.

**Strategy Insights:**
We found that the deep OTM and ITM calls weren't that profitable due to liquidity issues, so we mainly focused our strategies on ATM options where we could get better fills.

**Position Management:**
We implemented static delta hedging where we calculated delta at order time for position sizing. Each strike had its own position limits, and we tried to maintain delta neutral positions when possible for risk control.

**Results & Improvements:**
- Successfully implemented quantitative options pricing framework
- Effective volatility surface modeling for fair value estimation
- Potential improvements: 
  - **Dynamic Delta Hedging**: Recalculate delta at each iteration and adjust positions accordingly
  - **Enhanced Liquidity Filtering**: Better handling of illiquid deep ITM/OTM options
  - **Volatility Forecasting**: More sophisticated IV prediction models

---

## Round 4

### Strategy


### Results


---

## Round 5

### Strategy
At this point, we were trouble shooting our previous strategies so we did not focus on the new information of this round

### Results
Ultimately, there was some overfitting which happened which caused our Volcanic Rock strategy to a lose significant amount. If not for this loss, our team wouldve placed in the top 25. 