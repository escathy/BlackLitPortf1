# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 15:34:03 2021

@author: essta
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import pypfopt
from pypfopt import EfficientFrontier, objective_functions
from pypfopt import DiscreteAllocation

tickers = [ 
"NFRA",
"EWN",
"HDG",
"REET",
"EUDV",
"IJPE.L",
"D6RR.F",
"IPRV.AS",
"IEAC.AS",
"EUNH.F",
]

tickers1 = [
             "NFRA", #2
             "REET",
             "HDG",
             ]

ohlc = yf.download(tickers, period="10y")
prices = ohlc["Adj Close"]
prices.tail()

market_prices = yf.download("IEUR", period="10y")["Adj Close"]
market_prices.head()

mcaps = {}
for t in tickers1:
    stock = yf.Ticker(t)
    mcaps[t] = stock.info["totalAssets"]
mcaps

d = {"EUDV":57920000000, "IJPE.L":1022059200, "D6RR.F":83862781, "IPRV.AS":1240625100, "EWN": 281140000, "IEAC.AS": 10772260001, "EUNH.F": 4173217767}
mcaps.update(d)
##
z = yf.Ticker("IBCI.AS").info
###

from pypfopt import black_litterman, risk_models
from pypfopt import BlackLittermanModel, plotting

S = risk_models.CovarianceShrinkage(prices).ledoit_wolf()
delta = black_litterman.market_implied_risk_aversion(market_prices)

plotting.plot_covariance(S, plot_correlation=True);

market_prior = black_litterman.market_implied_prior_returns(mcaps, delta, S, risk_free_rate=0.003)
market_prior

market_prior.plot.barh(figsize=(10,5));

# You don't have to provide views on all the assets
# absolute views (i.e a return estimate for each asset)
viewdict = {

"NFRA": 0.082,

"EWN": 0.08,

"HDG": 0.062,

"REET": 0.064,

"EUDV": 0.08,

"IJPE.L": 0.063,

"D6RR.F": 0.08,

"IPRV.AS": 0.158,

"IEAC.AS":0.019,

"EUNH.F":0.012,

}


bl = BlackLittermanModel(S, pi=market_prior, absolute_views=viewdict)

confidences = [

0.6,

0.6,

0.5, # confident KO will do poorly

0.7,

0.6,

0.7,

0.6,

0.5,

0.7,

0.7

]


bl = BlackLittermanModel(S, pi=market_prior, absolute_views=viewdict, omega="idzorek", view_confidences=confidences)

######
fig, ax = plt.subplots(figsize=(7,7))
im = ax.imshow(bl.omega)
# We want to show all ticks...
ax.set_xticks(np.arange(len(bl.tickers)))
ax.set_yticks(np.arange(len(bl.tickers)))
ax.set_xticklabels(bl.tickers)
ax.set_yticklabels(bl.tickers)
plt.show()
######

np.diag(bl.omega)


intervals = [
(0.037, 0.179),

(0.038, 0.1213),

(0.033, 0.092),

(0.027, 0.102),

(0.038, 0.1213),

(0.0226, 0.1521),

(0.0382, 0.123),

(0.055, 0.273),

(0.007, 0.031),

(-0.004, 0.028),

]


variances = []
for lb, ub in intervals:
    sigma = (ub - lb)/2
    variances.append(sigma ** 2)

print(variances)
omega = np.diag(variances)

# We are using the shortcut to automatically compute market-implied prior
bl = BlackLittermanModel(S, pi="market", market_caps=mcaps, risk_aversion=delta,
                        absolute_views=viewdict, omega=omega)

# Posterior estimate of returns
ret_bl = bl.bl_returns()
ret_bl 

rets_df = pd.DataFrame([market_prior, ret_bl, pd.Series(viewdict)], 
             index=["Prior", "Posterior", "Views"]).T
rets_df

rets_df.plot.bar(figsize=(12,8));

S_bl = bl.bl_cov()
plotting.plot_covariance(S_bl);


###########Optimize
sector_mapper = {
    "NFRA": "ALT", 
    "EWN": "STOCKS", 
    "IEAC.AS": "BONDS",
    "HDG":  "ALT",
    "REET":  "ALT", 
    "EUDV": "STOCKS",  
    "IJPE.L": "STOCKS", 
    "D6RR.F": "ESG",
    "IPRV.AS": "ALT",
    "EUNH.F": "BONDS",
}

sector_lower = {
    "STOCKS": 0.25, # at least 30% to stocks
    "BONDS": 0.05 # at least 5% to bonds
    # For all other sectors, it will be assumed there is no lower bound
}

sector_upper = {
    "ALT": 0.25,
    "STOCKS": 0.5,
    "BONDS": 0.25,
    "ESG": 0.02
}

# we already using a Ledoit-Wolf shrinkage,
# which reduces the extreme values in the covariance matrix.

ef = EfficientFrontier(ret_bl, S_bl)
ef.add_sector_constraints(sector_mapper, sector_lower, sector_upper)
ef.add_objective(objective_functions.L2_reg, gamma=0.05)
ef.max_sharpe(risk_free_rate=0.003)
weights = ef.clean_weights()
weights

pd.Series(weights).plot.pie(figsize=(10,10));
da = DiscreteAllocation(weights, prices.iloc[-1], total_portfolio_value=30000000000)
alloc, leftover = da.lp_portfolio()
print(f"Leftover: ${leftover:.2f}")
alloc
ef.portfolio_performance(verbose=True, risk_free_rate=0.003)

num_small = len([k for k in weights if weights[k] <= 1e-4])
print(f"{num_small}/{len(ef.tickers)} tickers have zero weight")

# 
for sector in set(sector_mapper.values()):
    total_weight = 0
    for t,w in weights.items():
        if sector_mapper[t] == sector:
            total_weight += w
    print(f"{sector}: {total_weight:.3f}")


from pypfopt import expected_returns
from pypfopt import EfficientSemivariance

returns = expected_returns.returns_from_prices(prices).dropna()
returns.head()
mu = expected_returns.capm_return(prices)

portfolio_rets = (returns * weights).sum(axis=1)
portfolio_rets.hist(bins=100);

# VaR
var = portfolio_rets.quantile(0.05)
cvar = portfolio_rets[portfolio_rets <= var].mean()
print("VaR: {:.2f}%".format(100*var))
print("CVaR: {:.2f}%".format(100*cvar))



cov = risk_models.exp_cov(prices)
cor = risk_models.cov_to_corr(cov)
import seaborn as sns

mask = np.triu(np.ones_like(cor, dtype=bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(12, 12))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(cor, mask=mask, cmap=cmap, vmax=1, center=0,
            square=True, linewidths=.6, cbar_kws={"shrink": .6})
