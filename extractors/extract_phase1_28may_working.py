#!/usr/bin/env python3
"""
FILE: extract_phase1_28may_working.py
CREATED: 2026-05-28
CHAT: jmr_28may_v1
STATUS: Phase 1 extractor - core metrics only
PYMC: 5.26.1, ARVIZ: 0.22.0
PURPOSE: Extract ARI, parameter recovery, ESS, Rhat from all PKLs
"""
import pickle
import numpy as np
from pathlib import Path
from sklearn.metrics import adjusted_rand_score
from itertools import permutations
import pandas as pd
import arviz as az
import math
import warnings
import time
import argparse
import re
warnings.filterwarnings('ignore')

def best_ari(Z_pred, Z_true):
    K = int(max(Z_pred.max(), Z_true.max())) + 1
    best = -1
    for perm in permutations(range(K)):
        Z_perm = np.array([perm[z] for z in Z_pred.flatten()]).reshape(Z_pred.shape)
        ari = adjusted_rand_score(Z_true.flatten(), Z_perm.flatten())
        if ari > best:
            best = ari
    return float(best)

def forward_filter_ari(dgp, post, periods='train'):
    alpha_h_est = post['alpha_h'].mean(dim=['chain','draw']).values
    beta_m_est = post['beta_m'].mean(dim=['chain','draw']).values
    
    gamma_m_est = float(post['gamma_m'].mean(dim=['chain','draw']).values) if 'gamma_m' in post else 0.0
    
    if 'theta' in post:
        theta_raw = post['theta'].mean(dim=['chain','draw']).values
        theta_est = np.asarray(theta_raw).reshape(-1)
        if theta_est.size == 1:
            theta_est = np.full(dgp['N'], theta_est[0])
    else:
        theta_est = np.zeros(dgp['N'])
    
    Gamma_est = post['Gamma'].mean(dim=['chain','draw']).values if 'Gamma' in post else np.ones((3,3))/3.0
    pi0_est = post['pi0'].mean(dim=['chain','draw']).values if 'pi0' in post else np.ones(3)/3.0
    
    N = dgp['N']
    T = dgp['T']
    T_full = dgp['T_full']
    K = 3
    
    if periods == 'train':
        t_start, t_end = 0, T
    elif periods == 'holdout':
        t_start, t_end = T, T_full
    else:
        t_start, t_end = 0, T_full
    
    Y_timing = dgp['Y_timing'][:, t_start:t_end]
    Y_spend = dgp['Y_spend'][:, t_start:t_end]
    Z_true = dgp['Z_true'][:, t_start:t_end]
    
    Z_pred = np.zeros((N, t_end-t_start), dtype=int)
    
    for i in range(N):
        log_alpha = np.zeros((t_end-t_start, K))
        for t in range(t_end-t_start):
            y = Y_timing[i, t]
            z = Y_spend[i, t]
            for k in range(K):
                lam = np.exp(alpha_h_est[k] + theta_est[i])
                lam = max(lam, 1e-10)
                r_nb = 2.0
                p_zero = (r_nb / (r_nb + lam)) ** r_nb
                log_emit_timing = np.log(p_zero + 1e-10) if y == 0 else np.log(1 - p_zero + 1e-10)
                
                mu_spend = np.exp(beta_m_est[k] + gamma_m_est * theta_est[i])
                mu_spend = max(mu_spend, 1e-10)
                alpha_gamma = 2.0
                beta_gamma = alpha_gamma / mu_spend
                
                if z > 0:
                    log_emit_spend = ((alpha_gamma - 1) * np.log(z) 
                                      - beta_gamma * z 
                                      + alpha_gamma * np.log(beta_gamma) 
                                      - math.lgamma(alpha_gamma))
                else:
                    log_emit_spend = -100
                
                log_emit = log_emit_timing + log_emit_spend
                
                if t == 0:
                    log_alpha[t, k] = np.log(pi0_est[k] + 1e-10) + log_emit
                else:
                    trans_log = np.log(Gamma_est[:, k] + 1e-10)
                    log_alpha[t, k] = log_emit + np.max(log_alpha[t-1] + trans_log)
            
            Z_pred[i, t] = np.argmax(log_alpha[t])
    
    return best_ari(Z_pred, Z_true)

def gamma_diag_mae(dgp, post):
    Gamma_est = post['Gamma'].mean(dim=['chain','draw']).values
    Gamma_true = dgp['Gamma_true']
    if Gamma_est.shape == (3,3) and Gamma_true.shape == (3,3):
        diag_est = np.diag(Gamma_est)
        diag_true = np.diag(Gamma_true)
        return float(np.mean(np.abs(diag_est - diag_true)))
    return np.nan

def ess_by_class(idata, model_name):
    results = {}
    post = idata.posterior
    
    var_groups = {
        'theta': ['theta'],
        'alpha_h': ['alpha_h_raw'],
        'beta_m': ['beta_m_raw'],
        'Gamma': ['Gamma'],
        'pi0': ['pi0'],
        'gamma_m': ['gamma_m'],
        'r_nb': ['log_r']
    }
    
    if 'Heckman' in model_name:
        var_groups['rho'] = ['rho']
        var_groups['u'] = ['u']
        var_groups['v'] = ['v']
    elif 'Hurdle' in model_name:
        var_groups['u'] = ['u']
        var_groups['v'] = ['v']
    
    for group, vars_list in var_groups.items():
        try:
            available = [v for v in vars_list if v in post.data_vars]
            if not available:
                continue
            ess = az.ess(idata, var_names=available)
            vals = []
            for v in available:
                vals.extend(ess[v].values.flatten())
            results['ess_' + group] = float(np.nanmin(vals)) if vals else np.nan
        except:
            results['ess_' + group] = np.nan
    
    return results

def parse_filename(pkl_path):
    """Fallback: infer metadata from filename if not in PKL"""
    name = pkl_path.stem
    patterns = [
        (r'fit_(BEMMAOR-indiv-rfree|BEMMAOR-global-rfree|Heckman-rfree|Heckman-global-rfree|Hurdle-rfree|Hurdle-global-rfree)_(structural|independent|correlated|mixed)_N(\d+)_T(\d+)(_seed(\d+))?', 
         ['model_name', 'world', 'N', 'T', '_', 'seed'])
    ]
    
    for pattern, keys in patterns:
        m = re.match(pattern, name)
        if m:
            return {k: v for k, v in zip(keys, m.groups()) if not k.startswith('_')}
    
    return {}

def extract_single(pkl_path):
    row = {'pkl_path': str(pkl_path)}
    
    try:
        with open(pkl_path, 'rb') as f:
            fit = pickle.load(f)
        
        row['model_name'] = fit.get('model_name', 'unknown')
        row['world'] = fit.get('world', 'unknown')
        row['seed'] = fit.get('seed', -1)
        row['N'] = fit.get('N', -1)
        row['T'] = fit.get('T', -1)
        row['T_full'] = fit.get('T_full', -1)
        row['runtime_sec'] = fit.get('runtime_sec', np.nan)
        
        if row['model_name'] == 'unknown':
            parsed = parse_filename(pkl_path)
            row.update({k: v for k, v in parsed.items() if row.get(k) == 'unknown' or row.get(k) == -1})
        
        idata = fit['idata']
        post = idata.posterior
        
        with open(fit['dgp_path'], 'rb') as f:
            dgp = pickle.load(f)
        
        row['ari_train'] = forward_filter_ari(dgp, post, 'train')
        row['ari_holdout'] = forward_filter_ari(dgp, post, 'holdout')
        
        row['gamma_diag_mae'] = gamma_diag_mae(dgp, post)
        
        alpha_h_est = post['alpha_h'].mean(dim=['chain','draw']).values
        row['alpha_h_mae'] = float(np.mean(np.abs(alpha_h_est - dgp['alpha_h_true'])))
        
        beta_m_est = post['beta_m'].mean(dim=['chain','draw']).values
        row['beta_m_mae'] = float(np.mean(np.abs(beta_m_est - dgp['beta_m_true'])))
        
        if 'gamma_m' in post:
            gamma_m_est = float(post['gamma_m'].mean(dim=['chain','draw']).values)
            row['gamma_m_rec'] = gamma_m_est / dgp['gamma_m_true'] if dgp['gamma_m_true'] > 0 else gamma_m_est
            row['gamma_m_mae'] = abs(gamma_m_est - dgp['gamma_m_true'])
        else:
            row['gamma_m_rec'] = np.nan
            row['gamma_m_mae'] = np.nan
        
        if 'theta' in post:
            theta_est = post['theta'].mean(dim=['chain','draw']).values.squeeze()
            theta_true = dgp['theta_true']
            if theta_est.shape == theta_true.shape:
                row['theta_corr'] = float(np.corrcoef(theta_est, theta_true)[0, 1])
            else:
                row['theta_corr'] = np.nan
        else:
            row['theta_corr'] = np.nan
        
        if 'log_r' in post:
            r_est = np.exp(post['log_r'].mean(dim=['chain','draw']).values)
            r_true = dgp['r_nb_true']
            row['r_mae'] = float(np.mean(np.abs(r_est - r_true)))
        else:
            row['r_mae'] = np.nan
        
        if 'rho' in post:
            rho_est = float(post['rho'].mean(dim=['chain','draw']).values)
            row['rho_rec'] = rho_est / dgp['rho_true'] if dgp['rho_true'] != 0 else rho_est
            row['rho_mae'] = abs(rho_est - dgp['rho_true'])
        else:
            row['rho_rec'] = np.nan
            row['rho_mae'] = np.nan
        
        ess_results = ess_by_class(idata, row['model_name'])
        row.update(ess_results)
        
        try:
            rhat = az.rhat(idata)
            rhat_vals = []
            for v in rhat.data_vars.values():
                rhat_vals.extend(v.values.flatten())
            row['rhat_max'] = float(np.max(rhat_vals)) if rhat_vals else np.nan
        except:
            row['rhat_max'] = np.nan
        
        try:
            waic = az.waic(idata)
            row['waic'] = float(waic.elpd_waic)
        except:
            row['waic'] = np.nan
        
    except Exception as e:
        print('  ERROR on {}: {}: {}'.format(pkl_path.name, type(e).__name__, e))
        row['error'] = str(e)
    
    return row

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', type=str, default='/Users/sudhirvoleti/jmr_may/trials_28may')
    parser.add_argument('--output_csv', type=str, default='phase1_results_28may.csv')
    args = parser.parse_args()
    
    base = Path(args.base_dir)
    pkl_files = sorted(base.rglob('fit_*.pkl'))
    
    print('Found {} PKLs'.format(len(pkl_files)))
    
    results = []
    t0 = time.time()
    
    for i, pkl in enumerate(pkl_files):
        if i % 10 == 0:
            print('  [{}/{}] elapsed={:.0f}s'.format(i+1, len(pkl_files), time.time()-t0))
        
        row = extract_single(pkl)
        results.append(row)
    
    df = pd.DataFrame(results)
    out_path = base / args.output_csv
    df.to_csv(out_path, index=False)
    
    print('\nExtracted {} / {} PKLs'.format(len([r for r in results if 'error' not in r]), len(pkl_files)))
    print('Saved to: {}'.format(out_path))
    
    if len(df) > 0 and 'ari_train' in df.columns:
        print('\n=== QUICK SUMMARY ===')
        print(df.groupby('model_name')['ari_train'].agg(['count','mean','std']).round(4))
        print('\nBy world:')
        print(df.groupby('world')['ari_train'].agg(['count','mean']).round(4))
        print('\nBy model x world:')
        print(df.pivot_table(index='model_name', columns='world', values='ari_train', aggfunc='mean').round(4))

if __name__ == '__main__':
    main()
