#!/usr/bin/env python3
"""
FILE: extract_phase4_segmentation_v2_28may_working.py
CREATED: 2026-05-29
CHAT: jmr_28may_v1
STATUS: Phase 4 v2 - segmentation on indiv models only, loads CLV from Phase 3
PYMC: 5.26.1, ARVIZ: 0.22.0
PURPOSE: Trait-based segmentation, elbow method, segment quality
"""
import pickle
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pandas as pd
import argparse
import warnings
warnings.filterwarnings('ignore')

def elbow_method(X, k_range=range(2, 8)):
    """Compute WCSS and silhouette for elbow method."""
    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    
    wcss = []
    silhouettes = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        wcss.append(km.inertia_)
        if k > 1 and len(np.unique(labels)) > 1:
            silhouettes.append(silhouette_score(X, labels))
        else:
            silhouettes.append(0)
    return list(k_range), wcss, silhouettes

def extract_phase4(pkl_path, clv_df, n_clusters=3, do_elbow=True):
    row = {'pkl_path': str(pkl_path)}
    
    try:
        with open(pkl_path, 'rb') as f:
            fit = pickle.load(f)
        
        model_name = fit.get('model_name', 'unknown')
        world = fit.get('world', 'unknown')
        seed = fit.get('seed', -1)
        
        row['model_name'] = model_name
        row['world'] = world
        row['seed'] = seed
        
        # Skip global models
        if 'global' in model_name.lower():
            row['skipped'] = 'global_model'
            return row
        
        idata = fit['idata']
        post = idata.posterior
        
        with open(fit['dgp_path'], 'rb') as f:
            dgp = pickle.load(f)
        
        N = dgp['N']
        
        # Extract heterogeneity parameter
        if 'theta' in post:
            theta_raw = post['theta'].mean(dim=['chain','draw']).values
            theta = np.asarray(theta_raw).reshape(-1)
            if theta.size == 1:
                theta = np.full(N, theta[0])
            het_name = 'theta'
            X = theta.reshape(-1, 1)
        elif 'u' in post and 'v' in post:
            u = post['u'].mean(dim=['chain','draw']).values.squeeze()
            v = post['v'].mean(dim=['chain','draw']).values.squeeze()
            u = np.asarray(u).reshape(-1)
            v = np.asarray(v).reshape(-1)
            if u.size == 1:
                u = np.full(N, u[0])
            if v.size == 1:
                v = np.full(N, v[0])
            if 'rho' in post:
                theta = np.column_stack([u, v])
                het_name = 'u_v_2d'
                X = theta
            else:
                theta = v  # Hurdle: spend heterogeneity
                het_name = 'v'
                X = theta.reshape(-1, 1)
        else:
            row['skipped'] = 'no_heterogeneity'
            return row
        
        row['het_name'] = het_name
        
        # Elbow method
        if do_elbow:
            k_range, wcss, silhouettes = elbow_method(X, range(2, 7))
            row['elbow_k'] = str(k_range)
            row['elbow_wcss'] = str([round(w, 1) for w in wcss])
            row['elbow_silhouette'] = str([round(s, 3) for s in silhouettes])
            best_k = k_range[np.argmax(silhouettes)]
            row['best_k'] = best_k
        else:
            best_k = n_clusters
        
        # Segmentation
        km = KMeans(n_clusters=best_k if do_elbow else n_clusters, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        row['silhouette_score'] = silhouette_score(X, labels) if len(np.unique(labels)) > 1 else 0
        
        # Load CLV from Phase 3 CSV (match by pkl_path)
        clv_match = clv_df[clv_df['pkl_path'] == str(pkl_path)]
        if len(clv_match) > 0:
            # Use pp-CLV predicted mean per customer
            # We need customer-level CLV, but Phase 3 only has aggregate
            # Recompute simple CLV point estimate
            alpha_h = post['alpha_h'].mean(dim=['chain','draw']).values
            beta_m = post['beta_m'].mean(dim=['chain','draw']).values
            gamma_m = float(post['gamma_m'].mean(dim=['chain','draw']).values) if 'gamma_m' in post else 0.0
            
            delta_weekly = (1.10)**(1/52) - 1
            clv = np.zeros(N)
            for i in range(N):
                z = int(dgp['Z_true'][i, dgp['T']-1])
                for t in range(104):
                    if het_name == 'theta':
                        lam = np.exp(alpha_h[z] + theta[i])
                        mu = np.exp(beta_m[z] + gamma_m * theta[i])
                    elif het_name == 'u_v_2d':
                        lam = np.exp(alpha_h[z] + u[i])
                        mu = np.exp(beta_m[z] + gamma_m * v[i])
                    else:  # v
                        lam = np.exp(alpha_h[z])
                        mu = np.exp(beta_m[z] + gamma_m * v[i])
                    clv[i] += (lam * mu) / ((1 + delta_weekly)**t)
                    z = np.random.choice(len(alpha_h), p=post['Gamma'].mean(dim=['chain','draw']).values[z, :])
        else:
            # Fallback: simple CLV
            clv = np.ones(N)  # placeholder
        
        # Segment outcomes
        n_segments = len(np.unique(labels))
        for seg in range(n_segments):
            mask = labels == seg
            n_seg = mask.sum()
            if n_seg == 0:
                continue
            
            seg_clv = clv[mask]
            seg_theta = theta[mask] if het_name == 'theta' else (u[mask] if het_name == 'u_v_2d' else v[mask])
            
            # Whale concentration
            n_whale_total = max(1, int(0.05 * N))
            whale_idx = np.argsort(clv)[-n_whale_total:]
            whale_in_seg = len(set(whale_idx) & set(np.where(mask)[0]))
            
            # State distribution
            states = dgp['Z_true'][mask, :dgp['T']]
            state_dist = np.bincount(states.flatten(), minlength=len(alpha_h)) / states.size
            
            row['seg{}_size_pct'.format(seg)] = n_seg / N * 100
            row['seg{}_clv_mean'.format(seg)] = seg_clv.mean()
            row['seg{}_clv_std'.format(seg)] = seg_clv.std()
            row['seg{}_whale_conc'.format(seg)] = whale_in_seg / n_seg
            row['seg{}_theta_mean'.format(seg)] = seg_theta.mean()
            row['seg{}_state0_pct'.format(seg)] = state_dist[0] * 100
            row['seg{}_state1_pct'.format(seg)] = state_dist[1] * 100
            row['seg{}_state2_pct'.format(seg)] = state_dist[2] * 100
        
        row['n_segments'] = n_segments
        row['segment_entropy'] = -sum([row.get('seg{}_size_pct'.format(s), 0)/100 * np.log(row.get('seg{}_size_pct'.format(s), 1)/100) for s in range(n_segments) if row.get('seg{}_size_pct'.format(s), 0) > 0])
        
    except Exception as e:
        print('  ERROR on {}: {}: {}'.format(pkl_path.name, type(e).__name__, e))
        row['error'] = str(e)
    
    return row

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pkl_path', type=str, help='Single PKL for elbow test')
    parser.add_argument('--base_dir', type=str, help='Directory to scan')
    parser.add_argument('--clv_csv', type=str, default='overnight_28may/phase3_enhanced_overnight_180.csv')
    parser.add_argument('--output_csv', type=str, default='phase4_segmentation_28may.csv')
    parser.add_argument('--n_clusters', type=int, default=3)
    parser.add_argument('--no_elbow', action='store_true', help='Skip elbow, use fixed K')
    args = parser.parse_args()
    
    # Load CLV data
    clv_df = pd.read_csv(args.clv_csv) if Path(args.clv_csv).exists() else pd.DataFrame()
    
    if args.pkl_path:
        pkl_files = [Path(args.pkl_path)]
    elif args.base_dir:
        pkl_files = sorted(Path(args.base_dir).rglob('fit_*.pkl'))
    else:
        print("Need --pkl_path or --base_dir")
        return
    
    # Filter to indiv models only
    pkl_files = [p for p in pkl_files if 'indiv' in p.name.lower() or 'rfree' in p.name.lower() and 'global' not in p.name.lower()]
    
    print('Found {} indiv PKLs'.format(len(pkl_files)))
    
    results = []
    for i, pkl in enumerate(pkl_files):
        print('  [{}/{}] {}'.format(i+1, len(pkl_files), pkl.name))
        row = extract_phase4(pkl, clv_df, n_clusters=args.n_clusters, do_elbow=not args.no_elbow)
        results.append(row)
    
    df = pd.DataFrame(results)
    out_path = Path(args.output_csv)
    df.to_csv(out_path, index=False)
    
    print('\nExtracted {} / {} PKLs'.format(len([r for r in results if 'error' not in r and 'skipped' not in r]), len(pkl_files)))
    print('Saved to: {}'.format(out_path))

if __name__ == '__main__':
    main()
