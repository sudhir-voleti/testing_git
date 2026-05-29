#!/usr/bin/env python3
"""
FILE: model_bemmaor_global_27may.py
CREATED: 2026-05-27
CHAT: jmr_27mayv1
STATUS: BEMMAOR-global with shared theta + r_k free per state
PYMC: 5.26.1, ARVIZ: 0.22.0
PURPOSE: Ablation test - does individual theta_i matter for structural coupling?
MATCHES: dgp_rfree_27may_working.py
NEXT: Run 5 reps in parallel with BEMMAOR-indiv
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


def build_model(Y_timing, Y_spend, K=3):
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
        theta = pm.Normal('theta', mu=0, sigma=1)  # SHARED theta (scalar)
        gamma_h = 1.0
        gamma_m = pm.HalfNormal('gamma_m', sigma=1.0)

        log_r = pm.Normal('log_r', 0, 1, shape=K)
        r_nb = pt.exp(log_r)

        y_exp = Y_timing[..., None]
        z_exp = Y_spend[..., None]

        lam = pt.exp(alpha_h[None, None, :] + gamma_h * theta)
        lam_clipped = pt.clip(lam, 1e-10, 1e10)
        r_exp = r_nb[None, None, :]
        log_p_zero_nbd = r_exp * (pt.log(r_exp) - pt.log(r_exp + lam_clipped))
        log_p_pos = pt.log1p(-pt.exp(log_p_zero_nbd) + 1e-10)

        mu_spend = pt.exp(beta_m[None, None, :] + gamma_m * theta)
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


def fit_model(Y_timing, Y_spend, draws=500, chains=4, cores=4):
    model = build_model(Y_timing, Y_spend, K=3)
    with model:
        idata = pm.sample(
            draws=draws, chains=chains, cores=cores,
            tune=draws, target_accept=0.9,
            random_seed=42, progressbar=False
        )
    return idata


def _safe_ess_rhat(idata):
    ess_min = np.nan
    rhat_max = np.nan
    try:
        var_list = [v for v in idata.posterior.data_vars
                    if v not in ['log_likelihood', 'alpha_h', 'beta_m']]
        if not var_list:
            return ess_min, rhat_max
        ess = az.ess(idata, var_names=var_list)
        rhat = az.rhat(idata, var_names=var_list)
        ess_vals = []
        rhat_vals = []
        for v in ess.data_vars.values():
            if hasattr(v.values, 'flatten'):
                ess_vals.extend(v.values.flatten())
        for v in rhat.data_vars.values():
            if hasattr(v.values, 'flatten'):
                rhat_vals.extend(v.values.flatten())
        if ess_vals:
            ess_min = float(np.min(ess_vals))
        if rhat_vals:
            rhat_max = float(np.max(rhat_vals))
    except:
        pass
    return ess_min, rhat_max


def compute_ari_forward_filter(dgp, idata):
    post = idata.posterior

    alpha_h_est = post['alpha_h'].mean(dim=['chain', 'draw']).values
    beta_m_est = post['beta_m'].mean(dim=['chain', 'draw']).values

    if 'gamma_m' in post:
        gamma_m_est = float(post['gamma_m'].mean(dim=['chain', 'draw']).values)
    else:
        gamma_m_est = 0.0

    if 'theta' in post:
        theta_est = float(post['theta'].mean(dim=['chain', 'draw']).values)
    else:
        theta_est = 0.0

    if 'Gamma' in post:
        Gamma_est = post['Gamma'].mean(dim=['chain', 'draw']).values
    else:
        Gamma_est = np.ones((3, 3)) / 3.0

    if 'pi0' in post:
        pi0_est = post['pi0'].mean(dim=['chain', 'draw']).values
    else:
        pi0_est = np.ones(3) / 3.0

    Y_timing = dgp['Y_timing']
    Y_spend = dgp['Y_spend']
    N, T = Y_timing.shape
    K = 3

    Z_pred = np.zeros((N, T), dtype=int)

    for i in range(N):
        log_alpha = np.zeros((T, K))

        for t in range(T):
            y = Y_timing[i, t]
            z = Y_spend[i, t]

            for k in range(K):
                lam = np.exp(alpha_h_est[k] + theta_est)
                lam = max(lam, 1e-10)

                r_nb = 2.0
                p_zero = (r_nb / (r_nb + lam)) ** r_nb

                if y == 0:
                    log_emit_timing = np.log(p_zero + 1e-10)
                else:
                    log_emit_timing = np.log(1 - p_zero + 1e-10)

                mu_spend = np.exp(beta_m_est[k] + gamma_m_est * theta_est)
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

    return adjusted_rand_score(dgp['Z_true'].flatten(), Z_pred.flatten())


def extract_metrics(idata, dgp):
    metrics = {
        'model': 'BEMMAOR-global-rfree',
        'world': dgp['world'],
        'N': dgp['N'],
        'T': dgp['T'],
        'seed': dgp['seed']
    }

    ess_min, rhat_max = _safe_ess_rhat(idata)
    metrics['ess_min'] = ess_min
    metrics['rhat_max'] = rhat_max

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

    if 'gamma_m' in post:
        gamma_m_est = float(post['gamma_m'].mean(dim=['chain', 'draw']).values)
        gamma_m_true = dgp['gamma_m_true']
        metrics['gamma_m_rec'] = gamma_m_est / gamma_m_true if gamma_m_true > 0 else gamma_m_est
        metrics['gamma_m_mae'] = abs(gamma_m_est - gamma_m_true)
    else:
        metrics['gamma_m_rec'] = np.nan
        metrics['gamma_m_mae'] = np.nan

    if 'log_r' in post:
        r_est = np.exp(post['log_r'].mean(dim=['chain', 'draw']).values)
        r_true = dgp['r_nb_true']
        if np.isscalar(r_true):
            r_true = np.array([r_true] * 3)
        metrics['r_rec'] = float(np.mean(r_est / r_true))
        metrics['r_mae'] = float(np.mean(np.abs(r_est - r_true)))
    else:
        metrics['r_rec'] = np.nan
        metrics['r_mae'] = np.nan

    if 'theta' in post:
        theta_est = float(post['theta'].mean(dim=['chain', 'draw']).values)
        theta_true = dgp['theta_true']
        metrics['theta_corr'] = float(np.corrcoef(np.repeat(theta_est, dgp['N']), theta_true)[0, 1])
    else:
        metrics['theta_corr'] = np.nan

    try:
        metrics['ari'] = compute_ari_forward_filter(dgp, idata)
    except:
        metrics['ari'] = np.nan

    try:
        waic = az.waic(idata)
        metrics['waic'] = float(waic.elpd_waic)
    except:
        metrics['waic'] = np.nan

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dgp_path', type=str, required=True)
    parser.add_argument('--draws', type=int, default=500)
    parser.add_argument('--chains', type=int, default=4)
    parser.add_argument('--cores', type=int, default=4)
    parser.add_argument('--output_dir', type=str, default='outputs_27may')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.dgp_path, 'rb') as f:
        dgp = pickle.load(f)

    print('Fitting BEMMAOR-global (r-free) on ' + dgp['world'] + ' world...')
    t0 = time.time()

    try:
        idata = fit_model(dgp['Y_timing'], dgp['Y_spend'],
                         draws=args.draws, chains=args.chains, cores=args.cores)

        elapsed = time.time() - t0

        pkl_name = 'fit_BEMMAOR-global-rfree_' + dgp['world'] + '_N' + str(dgp['N']) + '_T' + str(dgp['T']) + '_seed' + str(dgp['seed']) + '.pkl'
        pkl_path = os.path.join(args.output_dir, pkl_name)
        with open(pkl_path, 'wb') as f:
            pickle.dump({
                'idata': idata,
                'dgp_path': args.dgp_path,
                'model_name': 'BEMMAOR-global-rfree',
                'world': dgp['world'],
                'seed': dgp['seed'],
                'N': dgp['N'],
                'T': dgp['T'],
                'T_full': dgp.get('T_full', dgp['T']),
                'runtime_sec': round(elapsed, 1),
                'timestamp': time.strftime('%Y%m%d_%H%M%S')
            }, f)


        metrics = extract_metrics(idata, dgp)
        metrics['runtime'] = round(elapsed, 1)
        metrics['pkl_path'] = pkl_path

        print('\n=== RESULTS ===')
        print('ARI:        ' + str(round(metrics['ari'], 3)))
        print('ESS min:    ' + str(int(metrics['ess_min'])))
        print('Rhat max:   ' + str(round(metrics['rhat_max'], 3)))
        print('theta_corr: ' + str(round(metrics['theta_corr'], 3)))
        print('gamma_m_rec:' + str(round(metrics['gamma_m_rec'], 3)))
        print('r_rec:      ' + str(round(metrics['r_rec'], 3)))
        print('alpha_h_mae:' + str(round(metrics['alpha_h_mae'], 3)))
        print('beta_m_mae: ' + str(round(metrics['beta_m_mae'], 3)))
        print('Runtime:    ' + str(round(elapsed, 1)) + 's')
        print('Saved PKL:  ' + pkl_path)

    except Exception as e:
        print('FAIL: ' + str(e))
        raise


if __name__ == '__main__':
    main()
