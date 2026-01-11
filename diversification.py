import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from scipy.stats import t as t_dist

def get_metrics(w, mu, cov, corr):
    vol = np.sqrt(w @ cov @ w)
    active = np.where(w > 1e-6)[0]
    avg_c = corr.iloc[active, active].where(np.triu(np.ones((len(active),)*2), 1).astype(bool)).stack().mean() if len(active) > 1 else 0
    return {'ret': w @ mu, 'vol': vol, 'sharpe': (w @ mu)/vol if vol > 0 else 0, 'en': 1/(w**2).sum(), 'avg_c': avg_c, 'n': len(active)}

def build_struct(edges, prices):
    rets = pd.DataFrame({f"{r.stock1}/{r.stock2}": (prices[r.stock1] - r.hedge_ratio * prices[r.stock2]).pct_change() for _, r in edges.iterrows()}).dropna()
    cov, corr = rets.cov().values, rets.corr()
    eig = np.linalg.eigvalsh(corr.values)
    eig = eig[eig > 1e-10]
    p = eig / eig.sum()
    return {'mu': edges['edge'].values, 'cov': cov, 'corr': corr, 'en_pca': np.exp(-np.sum(p * np.log(p + 1e-10))), 'names': rets.columns.tolist(), 'rets': rets}

def optimize_all(mu, cov, corr):
    results = {}
    # 1. Kelly
    vals = np.linalg.eigvalsh(cov)
    reg = cov + (max(1e-5, abs(vals.min()) + 1e-5) if (vals.max()/vals[vals>0].min()) > 1e6 else 0) * np.eye(len(mu))
    w_k = np.maximum(np.linalg.solve(reg, mu), 0)
    results['Kelly'] = w_k / w_k.sum() if w_k.sum() > 0 else np.ones(len(mu))/len(mu)
    
    # 2. Mean-Variance (Best of lambdas)
    best_w, best_s = None, -1
    for lam in [0.5, 1.0, 2.0, 5.0, 10.0]:
        res = minimize(lambda w: -(w @ mu - 0.5 * lam * (w @ cov @ w)), np.ones(len(mu))/len(mu), 
                       bounds=[(0, min(0.2, 0.95/20))]*len(mu), constraints={'type': 'eq', 'fun': lambda w: w.sum()-1})
        s = get_metrics(res.x, mu, cov, corr)['sharpe']
        if s > best_s: best_w, best_s = res.x, s
    results['MV'] = best_w

    # 3. Risk Parity
    w_rp = np.ones(len(mu))/len(mu)
    for _ in range(100):
        rc = w_rp * (cov @ w_rp) / np.sqrt(w_rp @ cov @ w_rp)
        w_rp = (w_rp * (np.sqrt(w_rp @ cov @ w_rp)/len(mu)) / (rc + 1e-10)); w_rp /= w_rp.sum()
    results['RP'] = w_rp

    # 4. HRP
    dist = np.sqrt((1 - corr) / 2)
    link = linkage(squareform(dist.values, checks=False), 'single')
    def get_quasi(l, n):
        sort_ix = [l[-1, 0], l[-1, 1]]
        while max(sort_ix) >= n:
            new_ix = []
            for x in sort_ix:
                if x >= n: new_ix.extend([l[int(x-n), 0], l[int(x-n), 1]])
                else: new_ix.append(x)
            sort_ix = new_ix
        return [int(x) for x in sort_ix]
    
    def bisect(cv, items):
        if len(items) == 1: return {items[0]: 1.0}
        l, r = items[:len(items)//2], items[len(items)//2:]
        v_l = np.ones(len(l)) @ cv[np.ix_(l, l)] @ np.ones(len(l)) / len(l)**2
        v_r = np.ones(len(r)) @ cv[np.ix_(r, r)] @ np.ones(len(r)) / len(r)**2
        w_l = 1 - v_l / (v_l + v_r)
        return {**{k: v * w_l for k, v in bisect(cv, l).items()}, **{k: v * (1-w_l) for k, v in bisect(cv, r).items()}}
    
    hrp_d = bisect(cov, get_quasi(link, len(mu)))
    results['HRP'] = np.array([hrp_d[i] for i in range(len(mu))])
    return results

def main():
    prices = pd.read_csv("data/sp500_prices_clean.csv", index_col='date', parse_dates=True)
    edges = pd.read_csv("data/pair_edges.csv").query("edge > 0").reset_index(drop=True)
    data = build_struct(edges, prices)
    
    # Pre-filtering if needed
    if len(edges) > 1000:
        sel, idxs = [], edges['edge'].sort_values(ascending=False).index
        for i in idxs:
            if len(sel) >= 500: break
            if not sel or all(abs(data['corr'].iloc[i, s]) < 0.6 for s in sel): sel.append(i)
        edges = edges.loc[sel].reset_index(drop=True)
        data = build_struct(edges, prices)

    # Solve & Filter
    results = optimize_all(data['mu'], data['cov'], data['corr'])
    best_name = max(results, key=lambda k: get_metrics(results[k], data['mu'], data['cov'], data['corr'])['sharpe'])
    w, m = results[best_name], get_metrics(results[best_name], data['mu'], data['cov'], data['corr'])
    
    t_crit = t_dist.ppf(0.975, len(data['rets'])-2)
    thresh = min(0.7, max(0.4 if m['en'] < 20 else 0.3, (t_crit / np.sqrt(len(data['rets'])-2+t_crit**2)) * (3 if m['en'] < 20 else 2)))
    
    final_sel = []
    # FIX: Track stocks used to avoid conflicts, not just pair correlations
    stocks_used = set()
    
    for i in np.argsort(-w):
        if w[i] <= 1e-6:
            continue
            
        # Get the stocks in this pair
        stock1 = edges.iloc[i]['stock1']
        stock2 = edges.iloc[i]['stock2']
        
        # Check if either stock is already used in another pair
        # This prevents the same stock from appearing in multiple pairs
        if stock1 in stocks_used or stock2 in stocks_used:
            continue
        
        # Check correlation with already selected pairs
        if final_sel and not all(abs(data['corr'].iloc[i, s]) < thresh for s in final_sel):
            continue
        
        # Add this pair
        final_sel.append(i)
        stocks_used.add(stock1)
        stocks_used.add(stock2)
    
    fw = np.zeros(len(w))
    fw[final_sel] = w[final_sel]
    fw /= fw.sum()
    
    # Save
    edges['weight'] = fw
    edges['capital'] = fw * 100000
    
    # Additional validation: verify no stock appears in multiple selected pairs
    selected_pairs = edges[edges['weight'] > 0]
    all_stocks = list(selected_pairs['stock1']) + list(selected_pairs['stock2'])
    if len(all_stocks) != len(set(all_stocks)):
        print("WARNING: Some stocks appear in multiple pairs!")
        stock_counts = pd.Series(all_stocks).value_counts()
        duplicates = stock_counts[stock_counts > 1]
        print(f"Duplicate stocks: {duplicates.to_dict()}")
    
    selected_pairs.to_csv("data/optimized_portfolio.csv", index=False)
    pd.DataFrame([get_metrics(fw, data['mu'], data['cov'], data['corr'])]).to_csv("data/portfolio_metrics.csv", index=False)
    print(f"Done. Best Method: {best_name}. Positions: {len(final_sel)}")
    print(f"Unique stocks: {len(stocks_used)} (should be {2*len(final_sel)})")

if __name__ == "__main__": main()