import numpy as np
import pymc as pm
import pytensor.tensor as pt
from pytensor import scan
import arviz as az
import pickle
import os
import argparse
import time
import warnings
from sklearn.metrics import adjusted_rand_score
import math
from scipy.stats import spearmanr

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
        fn=forward_step,
        sequences=[log_emit_seq],
        outputs_info=[log_alpha_norm_init, log_Z_init],
        non_sequences=[log_Gamma],
        strict=True
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

        # FIX 1: Sum-to-zero theta with tighter prior
        theta_raw = pm.Normal('theta_raw', mu=0, sigma=0.5, shape=(N, 1))
        theta = pm.Deterministic('theta', theta_raw - pt.mean(theta_raw))

        gamma_h = 1.0
        gamma_m = pm.HalfNormal('gamma_m', sigma=1.0)

        log_r = pm.Normal('log_r', 0, 1, shape=K)
        r_nb = pt.exp(log_r)

        alpha_gamma = 2.0

        y_exp = Y_timing[..., None]
        z_exp = Y_spend[..., None]

        lam = pt.exp(alpha_h[None, None, :] + gamma_h * theta[:, :, None])
        lam_clipped = pt.clip(lam, 1e-10, 1e10)
        r_exp = r_nb[None, None, :]
        log_p_zero_nbd = r_exp * (pt.log(r_exp) - pt.log(r_exp + lam_clipped))
        log_p_pos = pt.log1p(-pt.exp(log_p_zero_nbd) + 1e-10)

        mu_spend = pt.exp(beta_m[None, None, :] + gamma_m * theta[:, :, None])
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



def fit_model(Y_timing, Y_spend, K=3, draws=500, chains=4, cores=4,
              target_accept=0.9, random_seed=42):
    model = build_model(Y_timing, Y_spend, K=K)
    with model:
        idata = pm.sample(
            draws=draws,
            chains=chains,
            cores=cores,
            tune=draws,
            target_accept=target_accept,
            random_seed=random_seed,
            progressbar=True,
            return_inferencedata=True,
            idata_kwargs={"log_likelihood": True}
        )
    return idata, model



def extract_ess_rhat(idata):
    ess_min = np.nan
    rhat_max = np.nan
    try:
        # FIX 2: Include ALL vars including sorted deterministics
        var_list = [v for v in idata.posterior.data_vars
                    if v not in ['log_likelihood']]
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
    except Exception as e:
        print(" ESS/R-hat failed: " + str(e))

    return ess_min, rhat_max


def extract_hmc_diagnostics(idata):
    diagnostics = {}
    try:
        sample_stats = idata.sample_stats
        if 'step_size' in sample_stats:
            step_sizes = sample_stats.step_size.values.flatten()
            diagnostics['step_size_mean'] = float(np.mean(step_sizes))
            diagnostics['step_size_min'] = float(np.min(step_sizes))
        if 'tree_depth' in sample_stats:
            tree_depths = sample_stats.tree_depth.values.flatten()
            diagnostics['tree_depth_max'] = int(np.max(tree_depths))
            diagnostics['tree_depth_mean'] = float(np.mean(tree_depths))
        if 'diverging' in sample_stats:
            divergences = sample_stats.diverging.values.flatten()
            diagnostics['divergences'] = int(np.sum(divergences))
            diagnostics['divergence_rate'] = float(np.mean(divergences))
        if 'acceptance_rate' in sample_stats:
            acc_rates = sample_stats.acceptance_rate.values.flatten()
            diagnostics['acceptance_mean'] = float(np.mean(acc_rates))
            diagnostics['acceptance_min'] = float(np.min(acc_rates))
        if 'energy' in sample_stats:
            energies = sample_stats.energy.values.flatten()
            diagnostics['energy_mean'] = float(np.mean(energies))
            diagnostics['energy_std'] = float(np.std(energies))
        if 'energy_error' in sample_stats:
            energy_errors = sample_stats.energy_error.values.flatten()
            diagnostics['energy_error_max'] = float(np.max(np.abs(energy_errors)))
        if 'tree_depth' in sample_stats:
            max_depth = 10
            diagnostics['max_depth_rate'] = float(np.mean(tree_depths >= max_depth))
    except Exception as e:
        print(" HMC diagnostics failed: " + str(e))
    return diagnostics


def extract_posterior_geometry(idata, dgp):
    geometry = {}
    try:
        post = idata.posterior
        theta_samples = post['theta'].values
        theta_flat = theta_samples.reshape(-1, dgp['N'])
        gamma_m_samples = post['gamma_m'].values.flatten()
        theta_mean = np.mean(theta_flat, axis=0)
        if len(gamma_m_samples) > 1:
            theta_mean_samples = np.mean(theta_flat, axis=1)
            corr = np.corrcoef(theta_mean_samples, gamma_m_samples)[0, 1]
            geometry['theta_gamma_m_corr'] = float(corr)
        theta_subset = theta_flat[:, :min(10, dgp['N'])]
        param_matrix = np.column_stack([theta_subset, gamma_m_samples[:, None]])
        cov = np.cov(param_matrix.T)
        eigvals = np.linalg.eigvalsh(cov)
        geometry['eigval_min'] = float(np.min(eigvals))
        geometry['eigval_max'] = float(np.max(eigvals))
        geometry['eigval_ratio'] = float(np.max(eigvals) / (np.min(eigvals) + 1e-10))
        geometry['condition_number'] = float(np.linalg.cond(cov))
    except Exception as e:
        print("  Posterior geometry failed: " + str(e))
    return geometry


def compute_ari_forward_filter(dgp, idata):
    post = idata.posterior
    alpha_h_est = post['alpha_h'].mean(dim=['chain', 'draw']).values
    beta_m_est = post['beta_m'].mean(dim=['chain', 'draw']).values
    gamma_m_est = float(post['gamma_m'].mean(dim=['chain', 'draw']).values) if 'gamma_m' in post else 0.0
    theta_est = post['theta'].mean(dim=['chain', 'draw']).values.squeeze() if 'theta' in post else np.zeros(dgp['N'])
    Gamma_est = post['Gamma'].mean(dim=['chain', 'draw']).values if 'Gamma' in post else np.ones((dgp['K'], dgp['K'])) / dgp['K']
    pi0_est = post['pi0'].mean(dim=['chain', 'draw']).values if 'pi0' in post else np.ones(dgp['K']) / dgp['K']
    Y_timing = dgp['Y_timing']
    Y_spend = dgp['Y_spend']
    N, T = Y_timing.shape
    K = dgp['K']
    Z_true = dgp['Z_true']
    Z_pred = np.zeros((N, T), dtype=int)
    for i in range(N):
        log_alpha = np.zeros((T, K))
        for t in range(T):
            y = Y_timing[i, t]
            z = Y_spend[i, t]
            for k in range(K):
                lam = np.exp(alpha_h_est[k] + theta_est[i])
                lam = max(lam, 1e-10)
                r_nb = 2.0
                p_zero = (r_nb / (r_nb + lam)) ** r_nb
                if y == 0:
                    log_emit_timing = np.log(p_zero + 1e-10)
                else:
                    log_emit_timing = np.log(1 - p_zero + 1e-10)
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
    return adjusted_rand_score(Z_true.flatten(), Z_pred.flatten())


def extract_theta_coverage(idata, dgp):
    coverage = np.nan
    rmse = np.nan
    try:
        theta_post = idata.posterior.theta.values
        theta_post = theta_post.reshape(-1, dgp['N'])
        theta_true = dgp['theta_true']
        ci_lower = np.percentile(theta_post, 2.5, axis=0)
        ci_upper = np.percentile(theta_post, 97.5, axis=0)
        coverage = float(np.mean((theta_true >= ci_lower) & (theta_true <= ci_upper)))
        theta_mean = np.mean(theta_post, axis=0)
        rmse = float(np.sqrt(np.mean((theta_mean - theta_true)**2)))
    except Exception as e:
        print("  Theta coverage failed: " + str(e))
    return coverage, rmse



def extract_waic_loo(idata):
    waic = np.nan
    loo = np.nan
    try:
        if "log_likelihood" in idata.posterior:
            import xarray as xr
            idata_copy = idata.copy()
            ll_data = idata_copy.posterior["log_likelihood"]
            idata_copy.log_likelihood = xr.Dataset({"log_likelihood": ll_data})
            del idata_copy.posterior["log_likelihood"]
            waic_result = az.waic(idata_copy)
        else:
            waic_result = az.waic(idata)
        waic = float(waic_result.elpd_waic)
    except Exception as e:
        print("  WAIC failed: " + str(e))
    try:
        if "log_likelihood" in idata.posterior:
            import xarray as xr
            idata_copy = idata.copy()
            ll_data = idata_copy.posterior["log_likelihood"]
            idata_copy.log_likelihood = xr.Dataset({"log_likelihood": ll_data})
            del idata_copy.posterior["log_likelihood"]
            loo_result = az.loo(idata_copy)
        else:
            loo_result = az.loo(idata)
        loo = float(loo_result.elpd_loo)
    except Exception as e:
        print("  LOO failed: " + str(e))
    return waic, loo



def extract_bdt_metrics(idata, dgp):
    bdt = {}
    try:
        post = idata.posterior
        theta_est = post['theta'].mean(dim=['chain', 'draw']).values.squeeze()
        theta_true = dgp['theta_true']
        rank_corr, _ = spearmanr(theta_est, theta_true)
        bdt['theta_rank_corr'] = float(rank_corr)
        true_top10 = set(np.argsort(theta_true)[-int(0.1 * dgp['N']):])
        est_top10 = set(np.argsort(theta_est)[-int(0.1 * dgp['N']):])
        bdt['top10_precision'] = float(len(true_top10 & est_top10) / len(est_top10)) if est_top10 else 0.0
        bdt['top10_recall'] = float(len(true_top10 & est_top10) / len(true_top10)) if true_top10 else 0.0
        bdt['clv_rank_corr'] = float(rank_corr)
    except Exception as e:
        print("  BDT failed: " + str(e))
    return bdt


def extract_all_metrics(idata, dgp, model, runtime_sec):
    metrics = {
        'N': dgp['N'], 'T': dgp['T'], 'K': dgp['K'],
        'gamma_m_true': dgp['gamma_m'], 'pi0': dgp['pi0'], 'seed': dgp['seed'],
        'runtime_sec': runtime_sec,
    }
    ess_min, rhat_max = extract_ess_rhat(idata)
    metrics['ess_min'] = ess_min
    metrics['rhat_max'] = rhat_max
    try:
        ess_theta = az.ess(idata, var_names=["theta"])
        metrics['ess_theta_min'] = float(np.min(ess_theta.theta.values))
        metrics['ess_theta_median'] = float(np.median(ess_theta.theta.values))
    except:
        metrics['ess_theta_min'] = np.nan
        metrics['ess_theta_median'] = np.nan
    try:
        ess_gamma_m = az.ess(idata, var_names=["gamma_m"])
        metrics['ess_gamma_m'] = float(ess_gamma_m.gamma_m.values)
    except:
        metrics['ess_gamma_m'] = np.nan
    metrics.update(extract_hmc_diagnostics(idata))
    metrics.update(extract_posterior_geometry(idata, dgp))
    post = idata.posterior
    if 'gamma_m' in post:
        gamma_m_est = float(post['gamma_m'].mean(dim=['chain', 'draw']).values)
        gamma_m_true = dgp['gamma_m']
        metrics['gamma_m_bias'] = float(gamma_m_est - gamma_m_true)
        metrics['gamma_m_mae'] = float(abs(gamma_m_est - gamma_m_true))
        metrics['gamma_m_rec'] = gamma_m_est / gamma_m_true if gamma_m_true > 0 else gamma_m_est
    else:
        metrics['gamma_m_bias'] = np.nan
        metrics['gamma_m_mae'] = np.nan
        metrics['gamma_m_rec'] = np.nan
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
    if 'theta' in post:
        theta_est = post['theta'].mean(dim=['chain', 'draw']).values.squeeze()
        theta_true = dgp['theta_true']
        if theta_est.shape == theta_true.shape:
            metrics['theta_corr'] = float(np.corrcoef(theta_est, theta_true)[0, 1])
        else:
            metrics['theta_corr'] = np.nan
        cov, rmse = extract_theta_coverage(idata, dgp)
        metrics['theta_coverage'] = cov
        metrics['theta_rmse'] = rmse
    else:
        metrics['theta_corr'] = np.nan
        metrics['theta_coverage'] = np.nan
        metrics['theta_rmse'] = np.nan
    try:
        metrics['ari'] = compute_ari_forward_filter(dgp, idata)
    except Exception as e:
        print("  ARI failed: " + str(e))
        metrics['ari'] = np.nan
    waic, loo = extract_waic_loo(idata)
    metrics['waic'] = waic
    metrics['loo'] = loo
    metrics.update(extract_bdt_metrics(idata, dgp))
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dgp_path', type=str, required=True)
    parser.add_argument('--draws', type=int, default=500)
    parser.add_argument('--chains', type=int, default=4)
    parser.add_argument('--cores', type=int, default=4)
    parser.add_argument('--target_accept', type=float, default=0.9)
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading DGP: " + args.dgp_path)
    with open(args.dgp_path, 'rb') as f:
        dgp = pickle.load(f)

    print("Fitting: N=" + str(dgp['N']) + ", T=" + str(dgp['T']) + ", K=" + str(dgp['K']) + ", gamma_m=" + str(dgp['gamma_m']) + ", pi0=" + str(dgp['pi0']))

    t0 = time.time()
    try:
        idata, model = fit_model(
            dgp['Y_timing'], dgp['Y_spend'],
            K=dgp['K'],
            draws=args.draws,
            chains=args.chains,
            cores=args.cores,
            target_accept=args.target_accept,
            random_seed=dgp.get('seed', 42)
        )

        elapsed = time.time() - t0
        print("Sampling complete: " + str(round(elapsed, 1)) + "s")
        print("Extracting metrics...")
        metrics = extract_all_metrics(idata, dgp, model, elapsed)

        if args.verbose:
            print("=== METRICS ===")
            for k, v in metrics.items():
                if isinstance(v, float) and not np.isnan(v):
                    print("  " + k + ": " + str(round(v, 4)))
                else:
                    print("  " + k + ": " + str(v))

        pkl_name = "fit_jrssc_v2_N" + str(dgp['N']) + "_T" + str(dgp['T']) + "_K" + str(dgp['K']) + "_gm" + str(dgp['gamma_m']) + "_p0" + str(dgp['pi0']) + "_seed" + str(dgp['seed']) + ".pkl"
        pkl_path = os.path.join(args.output_dir, pkl_name)
        with open(pkl_path, 'wb') as f:
            pickle.dump({
                'idata': idata,
                'dgp': dgp,
                'metrics': metrics,
                'runtime_sec': elapsed,
                'timestamp': time.strftime('%Y%m%d_%H%M%S')
            }, f)
        print("Saved: " + pkl_path)
        print("Done.")

    except Exception as e:
        print("FAIL: " + str(e))
        import traceback
        traceback.print_exc()



if __name__ == '__main__':
    main()
