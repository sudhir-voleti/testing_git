#!/usr/bin/env python3
"""
FILE: dgp_rfm_28may_working.py
CREATED: 2026-05-28
CHAT: jmr_28may_v1
STATUS: DGP with T=52 + T+40=92 OOS + post-hoc RFM computation
PYMC: 5.26.1, ARVIZ: 0.22.0
PURPOSE: Generate T_full=92 periods, models train on T=52, OOS evaluates last 40
         RFM computed post-hoc from Y, standardized on training stats
NEXT: Unit test with BEMMAOR-indiv-rfree + RFM
"""
import numpy as np
import pickle
import os
import argparse


def compute_rfm_features(Y_timing, Y_spend, mask):
    """
    Compute time-varying RFM features from generated Y.
    R(t) = periods since last purchase (recency)
    F(t) = cumulative purchases up to t
    M(t) = cumulative spend / cumulative purchases (avg spend per purchase)
    """
    N, T = Y_timing.shape
    R = np.zeros((N, T), dtype=np.float32)
    F = np.zeros((N, T), dtype=np.float32)
    M = np.zeros((N, T), dtype=np.float32)
    
    for i in range(N):
        last_purchase = -1
        cum_freq = 0
        cum_spend = 0.0
        for t in range(T):
            if mask[i, t]:
                if Y_timing[i, t] > 0:
                    last_purchase = t
                    cum_freq += Y_timing[i, t]  # count purchases, not just binary
                    cum_spend += Y_spend[i, t]
                if last_purchase >= 0:
                    R[i, t] = t - last_purchase
                    F[i, t] = cum_freq
                    M[i, t] = cum_spend / cum_freq if cum_freq > 0 else 0.0
                else:
                    R[i, t] = t + 1  # no purchase yet
                    F[i, t] = 0
                    M[i, t] = 0.0
    
    return R, F, M


def standardize_rfm(R, F, M, train_T, mask):
    """
    Standardize RFM using training period stats only.
    Apply same standardization to full period (train + OOS).
    """
    # Training period only for stats
    R_train = R[:, :train_T][mask[:, :train_T]]
    F_train = F[:, :train_T][mask[:, :train_T]]
    M_train = M[:, :train_T][mask[:, :train_T]]
    
    # Means and stds from training
    R_mean, R_std = R_train.mean(), R_train.std() + 1e-6
    F_mean, F_std = F_train.mean(), F_train.std() + 1e-6
    M_mean, M_std = M_train.mean(), M_train.std() + 1e-6
    
    # Standardize full period
    R_scaled = (R - R_mean) / R_std
    F_scaled = (F - F_mean) / F_std
    M_scaled = (M - M_mean) / M_std
    
    stats = {
        'R_mean': float(R_mean), 'R_std': float(R_std),
        'F_mean': float(F_mean), 'F_std': float(F_std),
        'M_mean': float(M_mean), 'M_std': float(M_std)
    }
    
    return R_scaled, F_scaled, M_scaled, stats


def generate_dgp(world, N=100, T=52, seed=42):
    rng = np.random.RandomState(seed)
    T_full = T + 40  # 40 periods for OOS

    world_params = {
        'independent': {
            'Gamma': np.array([[0.90, 0.05, 0.05], [0.05, 0.90, 0.05], [0.05, 0.05, 0.90]]),
            'rho': 0.0, 'gamma_m_true': 0.0
        },
        'correlated': {
            'Gamma': np.array([[0.70, 0.20, 0.10], [0.15, 0.70, 0.15], [0.10, 0.20, 0.70]]),
            'rho': 0.4, 'gamma_m_true': 0.5
        },
        'mixed': {
            'Gamma': np.array([[0.80, 0.15, 0.05], [0.10, 0.80, 0.10], [0.05, 0.15, 0.80]]),
            'rho': 0.2, 'gamma_m_true': 0.3
        },
        'structural': {
            'Gamma': np.array([[0.85, 0.10, 0.05], [0.15, 0.70, 0.15], [0.05, 0.20, 0.75]]),
            'rho': 0.0, 'gamma_m_true': 0.5
        }
    }

    wp = world_params.get(world, world_params['structural'])
    Gamma = wp['Gamma']
    rho = wp['rho']
    gamma_m_true = wp['gamma_m_true']

    alpha_h = np.array([0.0, 1.0, 2.0])
    beta_m = np.array([0.0, 1.5, 3.0])
    shape_spend = 2.0
    gamma_h_true = 1.0
    theta = rng.normal(0, 1, size=N)

    r_nb = np.array([1.0, 2.0, 3.0])
    sigma_t = 0.3
    sigma_s = 0.5
    Sigma = [[sigma_t**2, rho*sigma_t*sigma_s], [rho*sigma_t*sigma_s, sigma_s**2]]
    errors = rng.multivariate_normal([0, 0], Sigma, size=(N, T_full))

    Z = np.zeros((N, T_full), dtype=int)
    for i in range(N):
        Z[i, 0] = rng.choice(3, p=[0.4, 0.4, 0.2])
        for t in range(1, T_full):
            Z[i, t] = rng.choice(3, p=Gamma[Z[i, t-1]])

    Y_timing = np.zeros((N, T_full), dtype=int)
    Y_spend = np.zeros((N, T_full))

    for i in range(N):
        for t in range(T_full):
            k = Z[i, t]
            log_lam = alpha_h[k] + gamma_h_true * theta[i] + errors[i, t, 0]
            lam = np.exp(log_lam)
            p_nb = r_nb[k] / (r_nb[k] + lam)
            Y_timing[i, t] = rng.negative_binomial(r_nb[k], p_nb)

            log_mu = beta_m[k] + gamma_m_true * theta[i] + errors[i, t, 1]
            mu_spend = np.exp(log_mu)
            scale = mu_spend / shape_spend
            Y_spend[i, t] = rng.gamma(shape=shape_spend, scale=scale)

    # Compute RFM from full Y
    mask = np.ones((N, T_full), dtype=bool)
    R_full, F_full, M_full = compute_rfm_features(Y_timing, Y_spend, mask)
    
    # Standardize using training stats only
    R_scaled, F_scaled, M_scaled, rfm_stats = standardize_rfm(R_full, F_full, M_full, T, mask)

    # Split into train and OOS
    R_train = R_scaled[:, :T]
    F_train = F_scaled[:, :T]
    M_train = M_scaled[:, :T]
    
    R_oos = R_scaled[:, T:]
    F_oos = F_scaled[:, T:]
    M_oos = M_scaled[:, T:]

    return {
        'N': N, 'T': T, 'T_full': T_full, 'T_oos': T_full - T,
        'seed': seed, 'world': world,
        'alpha_h_true': alpha_h, 'beta_m_true': beta_m,
        'shape_spend_true': shape_spend, 'r_nb_true': r_nb,
        'gamma_h_true': gamma_h_true, 'gamma_m_true': gamma_m_true,
        'theta_true': theta, 'rho_true': rho,
        'sigma_t_true': sigma_t, 'sigma_s_true': sigma_s,
        'Gamma_true': Gamma, 'Z_true': Z,
        'Y_timing': Y_timing, 'Y_spend': Y_spend,
        # RFM features
        'R': R_train, 'F': F_train, 'M': M_train,
        'R_oos': R_oos, 'F_oos': F_oos, 'M_oos': M_oos,
        'rfm_stats': rfm_stats,
    }


def print_dgp_stats(dgp):
    Y_timing = dgp['Y_timing']
    Y_spend = dgp['Y_spend']
    Z_true = dgp['Z_true']
    N, T_full = Y_timing.shape
    T = dgp['T']
    r_nb = dgp['r_nb_true']

    print('\n=== DGP DESCRIPTIVE STATISTICS ===')
    print('World:  ' + dgp['world'])
    print('N:      ' + str(N))
    print('T:      ' + str(T) + ' (training)')
    print('T_full: ' + str(T_full) + ' (including ' + str(T_full - T) + ' OOS periods)')

    # Training stats
    Y_timing_train = Y_timing[:, :T]
    Y_spend_train = Y_spend[:, :T]
    overall_sparsity = np.mean(Y_timing_train == 0)
    print('\n--- TRAINING PERIODS (1-' + str(T) + ') ---')
    print('Sparsity (timing=0):     ' + str(round(overall_sparsity * 100, 1)) + '%')
    print('Mean timing (all):       ' + str(round(np.mean(Y_timing_train), 2)))
    print('Mean timing (if >0):     ' + str(round(np.mean(Y_timing_train[Y_timing_train > 0]), 2)))
    print('Mean spend (all):        ' + str(round(np.mean(Y_spend_train), 2)))
    print('Mean spend (if >0):      ' + str(round(np.mean(Y_spend_train[Y_spend_train > 0]), 2)))

    # OOS stats
    Y_timing_oos = Y_timing[:, T:]
    Y_spend_oos = Y_spend[:, T:]
    oos_sparsity = np.mean(Y_timing_oos == 0)
    print('\n--- OOS PERIODS (' + str(T+1) + '-' + str(T_full) + ') ---')
    print('Sparsity (timing=0):     ' + str(round(oos_sparsity * 100, 1)) + '%')
    print('Mean timing (all):       ' + str(round(np.mean(Y_timing_oos), 2)))
    print('Mean spend (all):        ' + str(round(np.mean(Y_spend_oos), 2)))

    # RFM stats
    print('\n--- RFM FEATURES (training) ---')
    R = dgp['R']
    F = dgp['F']
    M = dgp['M']
    print('R: mean=' + str(round(R.mean(), 3)) + ' std=' + str(round(R.std(), 3)))
    print('F: mean=' + str(round(F.mean(), 3)) + ' std=' + str(round(F.std(), 3)))
    print('M: mean=' + str(round(M.mean(), 3)) + ' std=' + str(round(M.std(), 3)))

    print('\n--- STATE-SPECIFIC (training) ---')
    for k in range(3):
        mask = Z_true[:, :T] == k
        if mask.sum() == 0:
            continue
        y_t_k = Y_timing_train[mask]
        y_s_k = Y_spend_train[mask]
        sparsity_k = np.mean(y_t_k == 0)
        mean_t_k = np.mean(y_t_k)
        mean_s_k = np.mean(y_s_k)
        var_t_k = np.var(y_t_k)
        var_s_k = np.var(y_s_k)
        vmr_t = var_t_k / mean_t_k if mean_t_k > 0 else np.nan
        vmr_s = var_s_k / mean_s_k if mean_s_k > 0 else np.nan
        print('State ' + str(k) + ' (r=' + str(r_nb[k]) + '):')
        print('  Sparsity:    ' + str(round(sparsity_k * 100, 1)) + '%')
        print('  Mean timing: ' + str(round(mean_t_k, 2)) + ' (var/mean=' + str(round(vmr_t, 2)) + ')')
        print('  Mean spend:  ' + str(round(mean_s_k, 2)) + ' (var/mean=' + str(round(vmr_s, 2)) + ')')

    print('\n--- THETA DISTRIBUTION ---')
    theta = dgp['theta_true']
    print('Mean:  ' + str(round(np.mean(theta), 3)))
    print('Std:   ' + str(round(np.std(theta), 3)))
    print('Range: ' + str(round(np.min(theta), 3)) + ' to ' + str(round(np.max(theta), 3)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--world', type=str, default='structural')
    parser.add_argument('--N', type=int, default=100)
    parser.add_argument('--T', type=int, default=52)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default='outputs_28may')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    dgp = generate_dgp(args.world, N=args.N, T=args.T, seed=args.seed)
    print_dgp_stats(dgp)

    pkl_path = os.path.join(args.output_dir, 'dgp_' + args.world + '_N' + str(args.N) + '_T' + str(args.T) + '_seed' + str(args.seed) + '.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(dgp, f)
    print('\nSaved DGP to ' + pkl_path)


if __name__ == '__main__':
    main()
