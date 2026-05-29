#!/usr/bin/env python3
"""
FILE: extract_phase3_28may_working_v2.py
CREATED: 2026-05-28
CHAT: jmr_28may_v1
STATUS: Phase 3 extractor v2 - fixes scalar theta for global models
"""
import pickle
import numpy as np
from pathlib import Path
from itertools import permutations
from sklearn.metrics import adjusted_rand_score
import pandas as pd
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

def viterbi_decode(log_emission, Gamma, pi0):
    N, T, K = log_emission.shape
    Z = np.zeros((N, T), dtype=int)
    for i in range(N):
        log_delta = np.log(np.asarray(pi0).flatten() + 1e-12) + log_emission[i, 0]
        psi = np.zeros((T, K), dtype=int)
        for t in range(1, T):
            log_delta_t = np.zeros(K)
            for k in range(K):
                trans = log_delta + np.log(Gamma[:, k] + 1e-12)
                log_delta_t[k] = np.max(trans) + log_emission[i, t, k]
                psi[t, k] = np.argmax(trans)
            log_delta = log_delta_t
        Z[i, T-1] = np.argmax(log_delta)
        for t in range(T-2, -1, -1):
            Z[i, t] = psi[t+1, Z[i, t+1]]
    return Z

def compute_log_emission(dgp, alpha_h, beta_m, gamma_m, theta, r_nb, shape_spend, periods='all'):
    N = dgp['N']
    T = dgp['T']
    T_full = dgp['T_full']
    K = len(alpha_h)
    
    if periods == 'train':
        t_start, t_end = 0, T
    elif periods == 'oos':
        t_start, t_end = T, T_full
    else:
        t_start, t_end = 0, T_full
    
    Y_timing = dgp['Y_timing'][:, t_start:t_end]
    Y_spend = dgp['Y_spend'][:, t_start:t_end]
    T_use = t_end - t_start
    
    log_em = np.zeros((N, T_use, K))
    
    # Handle scalar theta
    theta_scalar = np.isscalar(theta) or (isinstance(theta, np.ndarray) and theta.size == 1)
    if theta_scalar:
        theta_val = float(np.asarray(theta).flatten()[0])
    else:
        theta = np.asarray(theta).flatten()
        if len(theta) != N:
            theta = np.full(N, theta[0]) if len(theta) > 0 else np.zeros(N)
    
    for k in range(K):
        if theta_scalar:
            lam = np.exp(alpha_h[k] + theta_val)
            lam = np.full(N, lam)
        else:
            lam = np.exp(alpha_h[k] + theta)
        
        lam = np.clip(lam, 1e-10, 1e10)
        p_zero = (r_nb[k] / (r_nb[k] + lam)) ** r_nb[k]
        
        for i in range(N):
            for t in range(T_use):
                y = Y_timing[i, t]
                z = Y_spend[i, t]
                
                if y == 0:
                    log_em_t = np.log(p_zero[i] + 1e-12)
                else:
                    log_em_t = np.log(1 - p_zero[i] + 1e-12)
                
                if theta_scalar:
                    mu_spend = np.exp(beta_m[k] + gamma_m * theta_val)
                else:
                    mu_spend = np.exp(beta_m[k] + gamma_m * theta[i])
                
                mu_spend = max(mu_spend, 1e-10)
                beta_gamma = shape_spend / mu_spend
                
                if z > 0:
                    log_em_s = ((shape_spend - 1) * np.log(z)
                                - beta_gamma * z
                                + shape_spend * np.log(beta_gamma)
                                - math.lgamma(shape_spend))
                else:
                    log_em_s = -100
                
                log_em[i, t, k] = log_em_t + log_em_s
    
    return log_em

def compute_lead_time(Z, target_state=2, dormant_state=0):
    N, T = Z.shape
    lead_times = []
    
    for i in range(N):
        t = 0
        while t < T:
            if Z[i, t] == dormant_state:
                exit_t = t
                while t < T and Z[i, t] == dormant_state:
                    t += 1
                if t >= T:
                    break
                
                entry_t = t
                while entry_t < T and Z[i, entry_t] != target_state:
                    entry_t += 1
                
                if entry_t < T:
                    lead_times.append(entry_t - exit_t)
            
            t += 1
    
    return np.array(lead_times) if lead_times else np.array([np.nan])

def compute_clv_single_draw(dgp, alpha_h, beta_m, gamma_m, theta, Gamma, pi0, r_nb, shape_spend, use_true_states=False, T_sim=104):
    N = dgp['N']
    T = dgp['T']
    K = len(alpha_h)
    
    # Handle scalar theta
    theta_scalar = np.isscalar(theta) or (isinstance(theta, np.ndarray) and theta.size == 1)
    if theta_scalar:
        theta_val = float(np.asarray(theta).flatten()[0])
    else:
        theta = np.asarray(theta).flatten()
        if len(theta) != N:
            theta = np.full(N, theta[0]) if len(theta) > 0 else np.zeros(N)
    
    if use_true_states:
        Z_init = dgp['Z_true'][:, T-1]
    else:
        log_em = compute_log_emission(dgp, alpha_h, beta_m, gamma_m, theta_val if theta_scalar else theta, r_nb, shape_spend, 'train')
        Z_viterbi = viterbi_decode(log_em, Gamma, pi0)
        Z_init = Z_viterbi[:, -1]
    
    delta_weekly = (1.10)**(1/52) - 1
    clv = np.zeros(N)
    
    for i in range(N):
        z = int(Z_init[i])
        for t in range(T_sim):
            if theta_scalar:
                lam = np.exp(alpha_h[z] + theta_val)
                mu_spend = np.exp(beta_m[z] + gamma_m * theta_val)
            else:
                lam = np.exp(alpha_h[z] + theta[i])
                mu_spend = np.exp(beta_m[z] + gamma_m * theta[i])
            
            expected_spend = lam * mu_spend
            clv[i] += expected_spend / ((1 + delta_weekly)**t)
            z = np.random.choice(K, p=Gamma[z, :])
    
    return clv

def extract_phase3(pkl_path, n_draws=100):
    row = {'pkl_path': str(pkl_path)}
    
    try:
        with open(pkl_path, 'rb') as f:
            fit = pickle.load(f)
        
        row['model_name'] = fit.get('model_name', 'unknown')
        row['world'] = fit.get('world', 'unknown')
        row['seed'] = fit.get('seed', -1)
        
        if row['model_name'] == 'unknown':
            name = pkl_path.stem
            m = re.match(r'fit_(BEMMAOR-indiv-rfree|BEMMAOR-global-rfree|Heckman-rfree|Heckman-global-rfree|Hurdle-rfree|Hurdle-global-rfree)_(structural|independent|correlated|mixed)_N\d+_T\d+(_seed\d+)?', name)
            if m:
                row['model_name'] = m.group(1)
                row['world'] = m.group(2)
        
        idata = fit['idata']
        post = idata.posterior
        
        with open(fit['dgp_path'], 'rb') as f:
            dgp = pickle.load(f)
        
        N = dgp['N']
        T = dgp['T']
        T_full = dgp['T_full']
        K = 3
        
        n_chains = post.dims.get('chain', 1)
        n_total_draws = post.dims.get('draw', 1)
        subsample_every = max(1, (n_chains * n_total_draws) // n_draws)
        
        alpha_h_all = post['alpha_h'].values
        beta_m_all = post['beta_m'].values
        gamma_m_all = post['gamma_m'].values if 'gamma_m' in post else np.zeros((n_chains, n_total_draws))
        log_r_all = post['log_r'].values if 'log_r' in post else np.zeros((n_chains, n_total_draws, K))
        Gamma_all = post['Gamma'].values if 'Gamma' in post else np.ones((n_chains, n_total_draws, K, K)) / K
        pi0_all = post['pi0'].values if 'pi0' in post else np.ones((n_chains, n_total_draws, K)) / K
        
        if 'theta' in post:
            theta_all = post['theta'].values
            if theta_all.ndim > 3:
                theta_all = theta_all.squeeze(-1)
        else:
            theta_all = np.zeros((n_chains, n_total_draws, N))
        
        shape_spend = dgp.get('shape_spend_true', 2.0)
        
        clv_pred_samples = []
        clv_true_samples = []
        
        count = 0
        for c in range(n_chains):
            for d in range(0, n_total_draws, subsample_every):
                if count >= n_draws:
                    break
                
                alpha_h = alpha_h_all[c, d]
                beta_m = beta_m_all[c, d]
                gamma_m = float(gamma_m_all[c, d]) if gamma_m_all.ndim > 2 else 0.0
                r_nb = np.exp(log_r_all[c, d]) if log_r_all.ndim > 2 else np.array([2.0]*K)
                Gamma = Gamma_all[c, d] if Gamma_all.ndim > 3 else np.eye(K)
                pi0 = pi0_all[c, d] if pi0_all.ndim > 2 else np.ones(K)/K
                theta = theta_all[c, d] if theta_all.ndim > 2 else np.zeros(N)
                
                clv_pred = compute_clv_single_draw(dgp, alpha_h, beta_m, gamma_m, theta, Gamma, pi0, r_nb, shape_spend, use_true_states=False)
                clv_true = compute_clv_single_draw(dgp, alpha_h, beta_m, gamma_m, theta, Gamma, pi0, r_nb, shape_spend, use_true_states=True)
                
                clv_pred_samples.append(clv_pred)
                clv_true_samples.append(clv_true)
                count += 1
        
        clv_pred_samples = np.array(clv_pred_samples)
        clv_true_samples = np.array(clv_true_samples)
        
        row['pp_clv_pred_mean'] = float(np.mean(clv_pred_samples))
        row['pp_clv_pred_std'] = float(np.std(clv_pred_samples))
        row['pp_clv_true_mean'] = float(np.mean(clv_true_samples))
        row['pp_clv_true_std'] = float(np.std(clv_true_samples))
        
        clv_pred_customer = np.mean(clv_pred_samples, axis=0)
        clv_true_customer = np.mean(clv_true_samples, axis=0)
        
        regret = clv_true_customer - clv_pred_customer
        row['regret_mean'] = float(np.mean(regret))
        row['regret_std'] = float(np.std(regret))
        row['regret_total'] = float(np.sum(regret))
        row['regret_pct'] = float(np.sum(regret) / np.sum(clv_true_customer) * 100) if np.sum(clv_true_customer) > 0 else 0
        
        misalloc = np.abs(clv_pred_customer - clv_true_customer)
        row['misalloc_mean'] = float(np.mean(misalloc))
        row['misalloc_total'] = float(np.sum(misalloc))
        row['misalloc_pct'] = float(np.sum(misalloc) / np.sum(clv_true_customer) * 100) if np.sum(clv_true_customer) > 0 else 0
        
        clv_pred_flat = clv_pred_samples.flatten()
        row['var95_pp'] = float(np.percentile(clv_pred_flat, 5))
        row['var99_pp'] = float(np.percentile(clv_pred_flat, 1))
        cvar95 = np.mean(clv_pred_flat[clv_pred_flat <= row['var95_pp']]) if np.any(clv_pred_flat <= row['var95_pp']) else row['var95_pp']
        cvar99 = np.mean(clv_pred_flat[clv_pred_flat <= row['var99_pp']]) if np.any(clv_pred_flat <= row['var99_pp']) else row['var99_pp']
        row['cvar95_pp'] = float(cvar95)
        row['cvar99_pp'] = float(cvar99)
        
        n_top20 = max(1, int(0.20 * N))
        pred_idx = np.argsort(clv_pred_customer)[-n_top20:]
        true_idx = np.argsort(clv_true_customer)[-n_top20:]
        row['target20_precision'] = len(set(pred_idx) & set(true_idx)) / n_top20
        row['target20_recall'] = len(set(pred_idx) & set(true_idx)) / n_top20
        
        n_whale = max(1, int(0.05 * N))
        pred_w_idx = np.argsort(clv_pred_customer)[-n_whale:]
        true_w_idx = np.argsort(clv_true_customer)[-n_whale:]
        row['whale_precision'] = len(set(pred_w_idx) & set(true_w_idx)) / n_whale
        row['whale_recall'] = len(set(pred_w_idx) & set(true_w_idx)) / n_whale
        row['whale_f1'] = 2 * row['whale_precision'] * row['whale_recall'] / (row['whale_precision'] + row['whale_recall']) if (row['whale_precision'] + row['whale_recall']) > 0 else 0
        
        # Lead times
        log_em_train = compute_log_emission(dgp, 
            post['alpha_h'].mean(dim=['chain','draw']).values,
            post['beta_m'].mean(dim=['chain','draw']).values,
            float(post['gamma_m'].mean(dim=['chain','draw']).values) if 'gamma_m' in post else 0.0,
            post['theta'].mean(dim=['chain','draw']).values.squeeze() if 'theta' in post else np.zeros(N),
            np.exp(post['log_r'].mean(dim=['chain','draw']).values) if 'log_r' in post else np.array([2.0]*K),
            shape_spend, 'train')
        
        Gamma_mean = post['Gamma'].mean(dim=['chain','draw']).values if 'Gamma' in post else np.eye(K)
        pi0_mean = post['pi0'].mean(dim=['chain','draw']).values if 'pi0' in post else np.ones(K)/K
        
        Z_pred_train = viterbi_decode(log_em_train, Gamma_mean, pi0_mean)
        lead_times_train = compute_lead_time(Z_pred_train, target_state=2, dormant_state=0)
        row['lead_time_train_mean'] = float(np.nanmean(lead_times_train)) if len(lead_times_train) > 0 else np.nan
        row['lead_time_train_std'] = float(np.nanstd(lead_times_train)) if len(lead_times_train) > 0 else np.nan
        row['lead_time_train_n'] = len(lead_times_train)
        
        lead_times_true = compute_lead_time(dgp['Z_true'][:, :T], target_state=2, dormant_state=0)
        row['lead_time_true_mean'] = float(np.nanmean(lead_times_true)) if len(lead_times_true) > 0 else np.nan
        row['lead_time_true_std'] = float(np.nanstd(lead_times_true)) if len(lead_times_true) > 0 else np.nan
        row['lead_time_true_n'] = len(lead_times_true)
        
        log_em_oos = compute_log_emission(dgp,
            post['alpha_h'].mean(dim=['chain','draw']).values,
            post['beta_m'].mean(dim=['chain','draw']).values,
            float(post['gamma_m'].mean(dim=['chain','draw']).values) if 'gamma_m' in post else 0.0,
            post['theta'].mean(dim=['chain','draw']).values.squeeze() if 'theta' in post else np.zeros(N),
            np.exp(post['log_r'].mean(dim=['chain','draw']).values) if 'log_r' in post else np.array([2.0]*K),
            shape_spend, 'oos')
        
        Z_pred_oos = viterbi_decode(log_em_oos, Gamma_mean, pi0_mean)
        lead_times_oos = compute_lead_time(Z_pred_oos, target_state=2, dormant_state=0)
        row['lead_time_oos_mean'] = float(np.nanmean(lead_times_oos)) if len(lead_times_oos) > 0 else np.nan
        row['lead_time_oos_std'] = float(np.nanstd(lead_times_oos)) if len(lead_times_oos) > 0 else np.nan
        row['lead_time_oos_n'] = len(lead_times_oos)
        
    except Exception as e:
        print('  ERROR on {}: {}: {}'.format(pkl_path.name, type(e).__name__, e))
        import traceback
        traceback.print_exc()
        row['error'] = str(e)
    
    return row

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', type=str, default='/Users/sudhirvoleti/jmr_may/trials_28may')
    parser.add_argument('--output_csv', type=str, default='phase3_results_28may_v2.csv')
    parser.add_argument('--n_draws', type=int, default=100)
    args = parser.parse_args()
    
    base = Path(args.base_dir)
    pkl_files = sorted(base.rglob('fit_*.pkl'))
    
    print('Found {} PKLs'.format(len(pkl_files)))
    print('Using {} draws per PKL'.format(args.n_draws))
    
    results = []
    t0 = time.time()
    
    for i, pkl in enumerate(pkl_files):
        print('  [{}/{}] {} elapsed={:.0f}s'.format(i+1, len(pkl_files), pkl.name, time.time()-t0))
        row = extract_phase3(pkl, n_draws=args.n_draws)
        results.append(row)
    
    df = pd.DataFrame(results)
    out_path = base / args.output_csv
    df.to_csv(out_path, index=False)
    
    print('\nExtracted {} / {} PKLs'.format(len([r for r in results if 'error' not in r]), len(pkl_files)))
    print('Saved to: {}'.format(out_path))
    
    if len(df) > 0 and 'pp_clv_pred_mean' in df.columns:
        print('\n=== QUICK SUMMARY ===')
        print('pp-CLV by model:')
        print(df.groupby('model_name')[['pp_clv_pred_mean', 'pp_clv_true_mean', 'regret_pct', 'misalloc_pct']].mean().round(2))
        print('\nLead time by model:')
        print(df.groupby('model_name')[['lead_time_train_mean', 'lead_time_true_mean']].mean().round(2))
        print('\nTargeting by model:')
        print(df.groupby('model_name')[['target20_precision', 'whale_f1']].mean().round(4))

if __name__ == '__main__':
    main()
