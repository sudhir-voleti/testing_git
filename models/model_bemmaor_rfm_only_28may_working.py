#!/usr/bin/env python3
"""
FILE: model_bemmaor_rfm_only_28may_working.py
CREATED: 2026-05-28
CHAT: jmr_28may_v1
STATUS: BEMMAOR with RFM ONLY (no theta_i) — unit test
PYMC: 5.26.1, ARVIZ: 0.22.0
PURPOSE: Test if RFM alone can replace theta_i for heterogeneity
"""
import numpy as np
import pymc as pm
import pytensor.tensor as pt
from pytensor import scan
import pickle
import os
import argparse
import time
import arviz as az
from sklearn.metrics import adjusted_rand_score
import math
import warnings
warnings.filterwarnings('ignore')


def forward_algorithm_scan(log_emission, log_Gamma, pi0):
    N, T, K = log_emission.shape
    log_alpha_init = pt.log(pi0)[None, :] + log_emission[:, 0, :]
    log_Z_init = pt.logsumexp(log_alpha_init, axis=1, keepdims=True)
    log_alpha_norm_init = log_alpha_init - log_Z_init

    def forward_step(log_emit_t, log_alpha_prev, log_Z_prev, log_Gamma):
        transition = log_alpha_prev[:, :, None] + log_Gamma[None, :, :]
        log_alpha_new = log_emit_t + pt.logsumexp(transition, axis=1)
        log_Z_t = pt.logsumexp(log_alpha_new, axis=1, keepdims=True)
        log_alpha_norm = log_alpha_new - log_Z_t
        return log_alpha_norm, log_Z_t

    log_emit_seq = log_emission[:, 1:, :].swapaxes(0, 1)
    (log_alpha_norm_seq, log_Z_seq), _ = scan(
        fn=forward_step, sequences=[log_emit_seq],
        outputs_info=[log_alpha_norm_init, log_Z_init],
        non_sequences=[log_Gamma], strict=True
    )
    log_marginal = log_Z_init.squeeze() + pt.sum(log_Z_seq.squeeze(), axis=0)
    return log_marginal


def build_model(Y_timing, Y_spend, K=3, R=None, F=None, M=None):
    N, T = Y_timing.shape
    with pm.Model() as model:
        Gamma = pm.Dirichlet('Gamma', a=np.ones(K), shape=(K, K))
        pi0 = pm.Dirichlet('pi0', a=np.ones(K))
        log_Gamma = pt.log(Gamma)

        alpha_h_raw = pm.Normal('alpha_h_raw', 0, 1, shape=K)
        alpha_h = pm.Deterministic('alpha_h', pt.sort(alpha_h_raw))
        beta_m_raw = pm.Normal('beta_m_raw', 0, 1, shape=K)
        beta_m = pm.Deterministic('beta_m', pt.sort(beta_m_raw))

        alpha_gamma = 2.0

        log_r = pm.Normal('log_r', 0, 1, shape=K)
        r_nb = pt.exp(log_r)

        y_exp = Y_timing[..., None]
        z_exp = Y_spend[..., None]

        alphaR_h = pm.Normal('alphaR_h', 0, 0.5, shape=K)
        alphaF_h = pm.Normal('alphaF_h', 0, 0.5, shape=K)
        betaM = pm.Normal('betaM', 0, 0.5, shape=K)

        R_exp = R[:, :, None]
        F_exp = F[:, :, None]
        M_exp = M[:, :, None]

        lam = pt.exp(alpha_h[None, None, :] + alphaR_h[None, None, :] * R_exp + alphaF_h[None, None, :] * F_exp)
        mu_spend = pt.exp(beta_m[None, None, :] + betaM[None, None, :] * M_exp)

        lam_clipped = pt.clip(lam, 1e-10, 1e10)
        r_exp = r_nb[None, None, :]
        log_p_zero_nbd = r_exp * (pt.log(r_exp) - pt.log(r_exp + lam_clipped))
        log_p_pos = pt.log1p(-pt.exp(log_p_zero_nbd) + 1e-10)

        beta_gamma = alpha_gamma / mu_spend
        z_clipped = pt.clip(z_exp, 1e-10, 1e10)
        log_gamma = ((alpha_gamma - 1) * pt.log(z_clipped)
                     - beta_gamma * z_exp
                     + alpha_gamma * pt.log(beta_gamma)
                     - pt.gammaln(alpha_gamma))

        log_emission = pt.where(pt.eq(y_exp, 0), log_p_zero_nbd, log_p_pos + log_gamma)
        logp_cust = forward_algorithm_scan(log_emission, log_Gamma, pi0)
        pm.Deterministic('log_likelihood', logp_cust)
        pm.Potential('loglike', pt.sum(logp_cust))
    return model


def fit_model(Y_timing, Y_spend, draws=500, chains=4, cores=4, R=None, F=None, M=None):
    model = build_model(Y_timing, Y_spend, K=3, R=R, F=F, M=M)
    with model:
        idata = pm.sample(
            draws=draws, chains=chains, cores=cores,
            tune=draws, target_accept=0.9,
            random_seed=42, progressbar=False
        )
    return idata


def compute_ari_forward_filter(dgp, idata):
    post = idata.posterior

    alpha_h_est = post['alpha_h'].mean(dim=['chain', 'draw']).values
    beta_m_est = post['beta_m'].mean(dim=['chain', 'draw']).values

    if 'Gamma' in post:
        Gamma_est = post['Gamma'].mean(dim=['chain', 'draw']).values
    else:
        Gamma_est = np.ones((3, 3)) / 3.0

    if 'pi0' in post:
        pi0_est = post['pi0'].mean(dim=['chain', 'draw']).values
    else:
        pi0_est = np.ones(3) / 3.0

    Y_timing = dgp['Y_timing']
    N, T_full = Y_timing.shape
    T_train = dgp['T']
    K = 3

    Y_timing_train = Y_timing[:, :T_train]
    Y_spend_train = dgp['Y_spend'][:, :T_train]
    Z_true_train = dgp['Z_true'][:, :T_train]

    Z_pred = np.zeros((N, T_train), dtype=int)

    for i in range(N):
        log_alpha = np.zeros((T_train, K))

        for t in range(T_train):
            y = Y_timing_train[i, t]
            z = Y_spend_train[i, t]

            for k in range(K):
                lam = np.exp(alpha_h_est[k])
                if 'R' in dgp:
                    R_it = dgp['R'][i, t]
                    F_it = dgp['F'][i, t]
                    if 'alphaR_h' in post:
                        alphaR_est = post['alphaR_h'].mean(dim=['chain', 'draw']).values
                        alphaF_est = post['alphaF_h'].mean(dim=['chain', 'draw']).values
                        lam = lam * np.exp(alphaR_est[k] * R_it + alphaF_est[k] * F_it)

                lam = max(lam, 1e-10)
                r_nb = 2.0
                p_zero = (r_nb / (r_nb + lam)) ** r_nb

                if y == 0:
                    log_emit_timing = np.log(p_zero + 1e-10)
                else:
                    log_emit_timing = np.log(1 - p_zero + 1e-10)

                mu_spend = np.exp(beta_m_est[k])
                if 'M' in dgp:
                    M_it = dgp['M'][i, t]
                    if 'betaM' in post:
                        betaM_est = post['betaM'].mean(dim=['chain', 'draw']).values
                        mu_spend = mu_spend * np.exp(betaM_est[k] * M_it)

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

    return adjusted_rand_score(Z_true_train.flatten(), Z_pred.flatten())


def extract_metrics(idata, dgp):
    metrics = {
        'model': 'BEMMAOR-RFM-only',
        'world': dgp['world'],
        'N': dgp['N'],
        'T': dgp['T'],
        'seed': dgp['seed']
    }

    post = idata.posterior

    if 'alpha_h' in post:
        alpha_h_est = post['alpha_h'].mean(dim=['chain', 'draw']).values
        metrics['alpha_h_mae'] = float(np.mean(np.abs(alpha_h_est - dgp['alpha_h_true'])))
    else:
        metrics['alpha_h_mae'] = np.nan

    if 'beta_m' in post:
        beta_m_est = post['beta_m'].mean(dim=['chain', 'draw']).values
        metrics['beta_m_mae'] = float(np.mean(np.abs(beta_m_est - dgp['beta_m_true'])))
    else:
        metrics['beta_m_mae'] = np.nan

    if 'alphaR_h' in post:
        alphaR_est = post['alphaR_h'].mean(dim=['chain', 'draw']).values
        metrics['alphaR_mean'] = float(np.mean(alphaR_est))
        metrics['alphaR_std'] = float(np.std(alphaR_est))

    if 'alphaF_h' in post:
        alphaF_est = post['alphaF_h'].mean(dim=['chain', 'draw']).values
        metrics['alphaF_mean'] = float(np.mean(alphaF_est))
        metrics['alphaF_std'] = float(np.std(alphaF_est))

    if 'betaM' in post:
        betaM_est = post['betaM'].mean(dim=['chain', 'draw']).values
        metrics['betaM_mean'] = float(np.mean(betaM_est))
        metrics['betaM_std'] = float(np.std(betaM_est))

    try:
        metrics['ari'] = compute_ari_forward_filter(dgp, idata)
    except:
        metrics['ari'] = np.nan

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dgp_path', type=str, required=True)
    parser.add_argument('--draws', type=int, default=500)
    parser.add_argument('--chains', type=int, default=4)
    parser.add_argument('--cores', type=int, default=4)
    parser.add_argument('--output_dir', type=str, default='outputs_28may')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.dgp_path, 'rb') as f:
        dgp = pickle.load(f)

    print('Fitting BEMMAOR-RFM-only on ' + dgp['world'] + ' world...')
    t0 = time.time()

    try:
        Y_timing_train = dgp['Y_timing'][:, :dgp['T']]
        Y_spend_train = dgp['Y_spend'][:, :dgp['T']]

        R = dgp['R'] if 'R' in dgp else None
        F = dgp['F'] if 'F' in dgp else None
        M = dgp['M'] if 'M' in dgp else None

        idata = fit_model(Y_timing_train, Y_spend_train,
                         draws=args.draws, chains=args.chains, cores=args.cores,
                         R=R, F=F, M=M)

        elapsed = time.time() - t0

        pkl_name = 'fit_BEMMAOR-RFM-only_' + dgp['world'] + '_N' + str(dgp['N']) + '_T' + str(dgp['T']) + '.pkl'
        pkl_path = os.path.join(args.output_dir, pkl_name)
        with open(pkl_path, 'wb') as f:
            pickle.dump({'idata': idata, 'dgp_path': args.dgp_path}, f)

        metrics = extract_metrics(idata, dgp)
        metrics['runtime'] = round(elapsed, 1)
        metrics['pkl_path'] = pkl_path

        print('\n=== RESULTS ===')
        print('ARI:        ' + str(round(metrics['ari'], 3)))
        print('alpha_h_mae:' + str(round(metrics['alpha_h_mae'], 3)))
        print('beta_m_mae: ' + str(round(metrics['beta_m_mae'], 3)))
        print('alphaR_mean:' + str(round(metrics.get('alphaR_mean', np.nan), 3)))
        print('alphaF_mean:' + str(round(metrics.get('alphaF_mean', np.nan), 3)))
        print('betaM_mean: ' + str(round(metrics.get('betaM_mean', np.nan), 3)))
        print('Runtime:    ' + str(round(elapsed, 1)) + 's')
        print('Saved PKL:  ' + pkl_path)

    except Exception as e:
        print('FAIL: ' + str(e))
        raise


if __name__ == '__main__':
    main()
