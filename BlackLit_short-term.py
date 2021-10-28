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

tickers = [  "NFRA", #2
             "EWN",  #6
             "FLOT",  #7
             "TBX", #8
             "HYZD", #9
             "HDG", #11
             "REET",
             "EUDV",
             "IJPE.L",
             "D6RR.F",
             "IPRV.AS",
             "IBCI.AS"
             ]

tickers1 = [
             "NFRA", #2
             "EWN",  #6
             "FLOT",  #7
             "TBX", #8
             "HYZD", #9
             "HDG", #11
             "REET"
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

d = {"EUDV":57920000000, "IJPE.L":1022059200, "D6RR.F":83862781, "IPRV.AS":1240625100, "IBCI.AS":1846032950, "REET": 3390120000 }
mcaps.update(d)
##
z = yf.Ticker("IBCI.AS").info
###

from pypfopt import black_litterman, risk_models
from pypfopt import BlackLittermanModel, plotting

S = risk_models.CovarianceShrinkage(prices).ledoit_wolf()
delta = black_litterman.market_implied_risk_aversion(market_prices)

plotting.plot_covariance(S, plot_correlation=True);

market_prior = black_litterman.market_implied_prior_returns(mcaps, delta, S, risk_free_rate=0.0)
market_prior

market_prior.plot.barh(figsize=(10,5));

# You don't have to provide views on all the assets
# absolute views (i.e a return estimate for each asset)
viewdict = {
    "NFRA": 0.061, 
    "EWN": 0.082, 
    "FLOT": 0.034,
    "TBX": 0.0307,
    "HYZD": 0.038,
    "HDG": 0.041,
    "REET": 0.06, 
    "EUDV": 0.082,  
    "IJPE.L": 0.051, 
    "D6RR.F": 0.064,
    "IPRV.AS": 0.064,
    "IBCI.AS": 0.023,
}


bl = BlackLittermanModel(S, pi=market_prior, absolute_views=viewdict)

confidences = [
    0.8,
    0.7,
    0.7,
    0.7,
    0.7, # confident in dominos
    0.7, # confident KO will do poorly
    0.5, 
    0.7,
    0.6,
    0.6,
    0.4,
    0.6,
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
    (-0.047, 0.169),
    (0.056, 0.109),
    (0.024, 0.044),
    (0.0207, 0.0407),
    (0.018, 0.048),
    (0.02, 0.082),
   (0.025, 0.096),
    (0.056, 0.109),
    (-0.0903, 0.1922),
    (0.041, 0.087),
    (-0.1103, 0.2383),
    (-0.01, 0.036),
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
    "FLOT": "BONDS",
    "TBX": "BONDS",
    "HYZD": "BONDS",
    "HDG":  "ALT",
    "REET":  "ALT", 
    "EUDV": "STOCKS",  
    "IJPE.L": "STOCKS", 
    "D6RR.F": "ESG",
    "IPRV.AS": "ALT",
    "IBCI.AS": "BONDS",
}

sector_lower = {
    "STOCKS": 0.25, # at least 30% to stocks
    "BONDS": 0.05 # at least 5% to bonds
    # For all other sectors, it will be assumed there is no lower bound
}

sector_upper = {
    "ALT": 0.35,
    "STOCKS": 0.515,
    "BONDS": 0.12,
    "ESG": 0.015
}

# we already using a Ledoit-Wolf shrinkage,
# which reduces the extreme values in the covariance matrix.

ef = EfficientFrontier(ret_bl, S_bl)
ef.add_sector_constraints(sector_mapper, sector_lower, sector_upper)
ef.add_objective(objective_functions.L2_reg, gamma=0.05)
ef.max_sharpe(risk_free_rate=0.0)
weights = ef.clean_weights()
weights

pd.Series(weights).plot.pie(figsize=(10,10));
da = DiscreteAllocation(weights, prices.iloc[-1], total_portfolio_value=30000000000)
alloc, leftover = da.lp_portfolio()
print(f"Leftover: ${leftover:.2f}")
alloc
ef.portfolio_performance(verbose=True, risk_free_rate=0.0)

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
cov = risk_models.exp_cov(prices)
cor = risk_models.cov_to_corr(cov)

portfolio_rets = (returns * weights).sum(axis=1)
portfolio_rets.hist(bins=100);

# VaR
var = portfolio_rets.quantile(0.05)
cvar = portfolio_rets[portfolio_rets <= var].mean()
print("VaR: {:.2f}%".format(100*var))
print("CVaR: {:.2f}%".format(100*cvar))



pypfopt.plotting.plot_covariance(S, plot_correlation=(True))
import seaborn as sns

mask = np.triu(np.ones_like(cor, dtype=bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(12, 12))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(cor, mask=mask, cmap=cmap, vmax=1, center=0,
            square=True, linewidths=.6, cbar_kws={"shrink": .6})