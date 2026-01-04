import pandas as pd, numpy as np, os
from scipy.optimize import linprog
from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest
from auth import KEY, SECRET
from tp_sl import TAKE_PROFIT_Z, STOP_LOSS_Z

# Setup
tc, dc = TradingClient(KEY, SECRET, paper=True), StockHistoricalDataClient(KEY, SECRET)
acct = float(tc.get_account().equity)
held = {p.symbol for p in tc.get_all_positions()}

# Load Data
df = pd.read_csv("data/entry_signals.csv")
px = pd.read_csv("data/sp500_prices_clean.csv", index_col='date', parse_dates=True)

# 1. Calc Win Prob (Global)
wp = 0.51
if os.path.exists("data/trade_history.csv"):
    try:
        h = pd.read_csv("data/trade_history.csv")
        c = h[h['exit_date'].notna() & h['win'].notna()]
        if len(c) >= 10: wp = 0.6 * c['win'].mean() + 0.4 * 0.51
    except: pass

# 2. Kelly Calc with Annualized Returns
res = []
TRADING_DAYS_PER_YEAR = 252

for _, r in df.iterrows():
    if r['stock1'] in held or r['stock2'] in held: continue
    
    # Skip if half-life is invalid
    if pd.isna(r['half_life']) or r['half_life'] <= 0 or np.isinf(r['half_life']):
        continue
    
    s = px[r['stock1']] - r['hedge_ratio'] * px[r['stock2']]
    d_p, d_l = (abs(r['z_score'] - TAKE_PROFIT_Z), abs(STOP_LOSS_Z - r['z_score'])) if r['z_score'] > 0 else (abs(r['z_score'] + TAKE_PROFIT_Z), abs(-STOP_LOSS_Z - r['z_score']))
    exp_p, exp_l = d_p * s.std(), d_l * s.std()
    wl = exp_p / exp_l if exp_l > 0 else 1.0
    
    # Expected return per trade
    per_trade_return = wp * wl - (1 - wp)  # This is the edge
    
    # Expected number of trades per year (assuming each trade takes ~half_life days to resolve)
    trades_per_year = TRADING_DAYS_PER_YEAR / r['half_life']
    
    # Annualized expected return (compounded)
    # Using simplified: annualized_return ≈ per_trade_return * trades_per_year
    # For more accuracy with compounding: (1 + per_trade_return)^trades_per_year - 1
    annualized_return = per_trade_return * trades_per_year
    
    # Kelly fraction based on single-trade metrics
    raw_k = (wp * wl - (1 - wp)) / wl if wl > 0 else 0
    
    # Apply conservative scaling and caps
    k = min(max(0, raw_k * 0.33), 0.08)
    
    if k > 0.001 and annualized_return > 0: 
        res.append({
            **r, 
            'win_prob': wp, 
            'win_loss_ratio': wl,
            'per_trade_return': per_trade_return,
            'trades_per_year': trades_per_year,
            'annualized_return': annualized_return,
            'raw_kelly_fraction': k
        })

df = pd.DataFrame(res)
if df.empty:
    pd.DataFrame(columns=['stock1','stock2','signal','z_score','hedge_ratio','kelly_fraction','capital_allocation','shares1','shares2','price1','price2','win_prob','win_loss_ratio','half_life','annualized_return']).to_csv("data/sized_signals.csv", index=False)
    exit()

# 3. Weight by annualized return for capital allocation
# Higher annualized returns get proportionally more capital
df['return_weight'] = df['annualized_return'] / df['annualized_return'].sum()

# Target capital based on both Kelly fraction and annualized return
# Blend: 50% Kelly-based, 50% return-weighted
tgt_cap = acct * 0.95
kelly_weight = df['raw_kelly_fraction'] / df['raw_kelly_fraction'].sum()
df['combined_weight'] = 0.5 * kelly_weight + 0.5 * df['return_weight']
df['target_capital'] = df['combined_weight'] * tgt_cap

snapped = []

for _, r in df.iterrows():
    try:
        q = dc.get_stock_latest_quote(StockLatestQuoteRequest(symbol_or_symbols=[r['stock1'], r['stock2']]))
        p1, p2 = (q[r['stock1']].bid_price + q[r['stock1']].ask_price)/2, (q[r['stock2']].bid_price + q[r['stock2']].ask_price)/2
    except: continue
    
    feas = []
    min_c, max_c = 0.01 * acct, 0.08 * acct
    s1_min, s1_max = max(1, int(min_c / (p1 * (1 + 1/abs(r['hedge_ratio']))))), int(max_c / p1) + 10
    
    for s1 in range(s1_min, min(s1_max, 500)):
        ts2 = (s1 * p1) / (abs(r['hedge_ratio']) * p2)
        for s2 in [int(ts2), int(ts2)+1]:
            if s2 < 1: continue
            cap = s1*p1 + s2*p2
            if abs(s1*p1/(s2*p2) - abs(r['hedge_ratio']))/abs(r['hedge_ratio'])*100 <= 5.0 and min_c <= cap <= max_c:
                if not any(abs(f['capital'] - cap) < 10 for f in feas):
                    feas.append({'shares1': s1, 'shares2': s2, 'capital': cap, 'pct': cap/acct, 'hedge_error': abs(s1*p1/(s2*p2) - abs(r['hedge_ratio']))/abs(r['hedge_ratio'])*100})
    
    if feas:
        best = min(feas, key=lambda x: abs(x['capital'] - r['target_capital']))
        snapped.append({**r, 'price1': p1, 'price2': p2, 'feasible': feas, 'snap_cap': best['capital'], 'snap_s1': best['shares1'], 'snap_s2': best['shares2'], 'snap_pct': best['pct'], 'snap_err': best['hedge_error']})

sdf = pd.DataFrame(snapped)
if sdf.empty: exit()

# 4. Optimization (LP) or Fallback
final = []
if sdf['snap_cap'].sum() <= tgt_cap:
    final = [dict(row, shares1=row['snap_s1'], shares2=row['snap_s2'], capital_allocation=row['snap_cap'], kelly_fraction=row['snap_pct'], hedge_error=row['snap_err']) for _, row in sdf.iterrows()]
else:
    vars_ = []
    for i, r in sdf.iterrows():
        for j, alloc in enumerate(r['feasible']):
            vars_.append({'pid': i, 'cap': alloc['capital'], 'alloc': alloc, 'row': r})
            
    c = [-1.0/len(sdf.loc[v['pid']]['feasible']) for v in vars_]
    A_ub = [[v['cap'] for v in vars_]] + [[1 if v['pid']==pid else 0 for v in vars_] for pid in sdf.index]
    b_ub = [tgt_cap] + [1] * len(sdf)
    
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=(0,1), method='highs')
    
    if res.success:
        sel_idx, budget = set(), tgt_cap
        cands = sorted(enumerate(res.x), key=lambda x: x[1], reverse=True)
        for i, w in cands:
            v = vars_[i]
            if v['pid'] not in sel_idx and v['cap'] <= budget:
                final.append({**v['row'], 'shares1': v['alloc']['shares1'], 'shares2': v['alloc']['shares2'], 'capital_allocation': v['alloc']['capital'], 'kelly_fraction': v['alloc']['pct'], 'hedge_error': v['alloc']['hedge_error']})
                sel_idx.add(v['pid'])
                budget -= v['cap']
        
        if budget > acct * 0.02:
            for i, r in sdf.iterrows():
                if i not in sel_idx:
                    valid = [f for f in r['feasible'] if f['capital'] <= budget]
                    if valid:
                        best = max(valid, key=lambda x: x['capital'])
                        final.append({**r, 'shares1': best['shares1'], 'shares2': best['shares2'], 'capital_allocation': best['capital'], 'kelly_fraction': best['pct'], 'hedge_error': best['hedge_error']})
                        budget -= best['capital']
    else:
        budget = tgt_cap
        for _, r in sdf.sort_values('annualized_return', ascending=False).iterrows():
            if r['snap_cap'] <= budget:
                final.append({**r, 'shares1': r['snap_s1'], 'shares2': r['snap_s2'], 'capital_allocation': r['snap_cap'], 'kelly_fraction': r['snap_pct'], 'hedge_error': r['snap_err']})
                budget -= r['snap_cap']

# Output
out = pd.DataFrame(final)
if not out.empty:
    out = out.sort_values('annualized_return', ascending=False)
    out[['stock1', 'stock2', 'signal', 'z_score', 'hedge_ratio', 'half_life', 'annualized_return', 'kelly_fraction', 'capital_allocation', 'shares1', 'shares2', 'price1', 'price2', 'win_prob', 'win_loss_ratio']].to_csv("data/sized_signals.csv", index=False)
    print(f"Saved {len(out)} signals. Target: ${tgt_cap:,.0f}, Alloc: ${out['capital_allocation'].sum():,.0f}")
    print(f"Avg annualized return: {out['annualized_return'].mean():.2%}")