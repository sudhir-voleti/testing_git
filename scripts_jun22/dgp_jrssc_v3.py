#!/usr/bin/env python3
"""
DGP v3: Supports multiple coupling regimes for misspecification robustness
Regimes: pure_theta, pure_rho, mixed, none
"""
import numpy as np
import pickle
import os
import argparse

def generate_dgp_jrssc(N=100, T=52, K=3, gamma_m=0.5, pi0=0.8, seed=42,
                        alpha_gamma=2.0, state_sep='baseline', sigma_theta=1.0,
                        regime='pure_theta'):
    """
    regime: 'pure_theta' -> gamma_m active, rho=0
            'pure_rho' -> gamma_m=0, rho active (residual correlation)
            'mixed' -> both gamma_m and rho active
            'none' -> gamma_m=0, rho=0 (independent)
    """
    rng = np.random.RandomState(seed)
    
    # Set regime-specific parameters
    if regime == 'pure_theta':
        gamma_m_eff = gamma_m
        rho = 0.0
    elif regime == 'pure_rho':
        gamma_m_eff = 0.0
        rho = 0.5
    elif regime == 'mixed':
        gamma_m_eff = gamma_m
        rho = 0.5
    elif regime == 'none':
        gamma_m_eff = 0.0
        rho = 0.0
    else:
        raise ValueError("regime must be one of: pure_theta, pure_rho, mixed, none")
    
    theta_true = rng.normal(0, sigma_theta, size=N)
    
    Gamma_true = np.eye(K) * 0.7 + np.ones((K, K)) * 0.1
    Gamma_true = Gamma_true / Gamma_true.sum(axis=1, keepdims=True)
    pi0_vec = np.ones(K) / K
    
    # State separation
    if state_sep == 'close':
        alpha_h_true = np.linspace(-0.5, 0.5, K)
        beta_m_true = np.linspace(2.5, 3.5, K)
    elif state_sep == 'far':
        alpha_h_true = np.linspace(-2.0, 2.0, K)
        beta_m_true = np.linspace(1.0, 5.0, K)
    else:  # baseline
        alpha_h_true = np.linspace(-1.0, 1.0, K)
        beta_m_true = np.linspace(2.0, 4.0, K)
    
    shape_spend_true = alpha_gamma
    r_nb_true = np.ones(K) * 2.0
    gamma_h_true = 1.0
    
    Z_true = np.zeros((N, T), dtype=int)
    for i in range(N):
        Z_true[i, 0] = rng.choice(K, p=pi0_vec)
        for t in range(1, T):
            Z_true[i, t] = rng.choice(K, p=Gamma_true[Z_true[i, t-1], :])
    
    Y_timing = np.zeros((N, T), dtype=int)
    Y_spend = np.zeros((N, T))
    
    for i in range(N):
        for t in range(T):
            z = Z_true[i, t]
            
            # Generate correlated errors for residual correlation
            if rho != 0:
                mean = [0, 0]
                cov = [[0.3**2, rho * 0.3 * 0.5], [rho * 0.3 * 0.5, 0.5**2]]
                eps_timing, eps_spend = rng.multivariate_normal(mean, cov)
            else:
                eps_timing = rng.normal(0, 0.3)
                eps_spend = rng.normal(0, 0.5)
            
            # Timing: NB with possible theta coupling
            log_lam = alpha_h_true[z] + gamma_h_true * theta_true[i] + gamma_m_eff * theta_true[i] + eps_timing
            lam = np.exp(log_lam)
            lam = max(lam, 1e-10)
            r = r_nb_true[z]
            p = r / (r + lam)
            Y_timing[i, t] = rng.negative_binomial(r, p)
            
            # Spend: Gamma with possible theta coupling
            log_mu = beta_m_true[z] + gamma_m_eff * theta_true[i] + eps_spend
            mu_spend = np.exp(log_mu)
            mu_spend = max(mu_spend, 1e-10)
            scale = mu_spend / shape_spend_true
            Y_spend[i, t] = rng.gamma(shape_spend_true, scale=scale)
    
    # Apply sparsity mask
    purchase_mask = rng.binomial(1, pi0, size=(N, T))
    Y_spend = Y_spend * purchase_mask
    Y_timing = Y_timing * purchase_mask
    
    sparsity = float(np.mean(Y_timing == 0))
    
    dgp = {
        'N': N, 'T': T, 'K': K,
        'gamma_m': gamma_m_eff, 'pi0': pi0, 'seed': seed,
        'alpha_gamma': alpha_gamma, 'state_sep': state_sep,
        'sigma_theta': sigma_theta, 'regime': regime, 'rho': rho,
        'theta_true': theta_true, 'Gamma_true': Gamma_true,
        'Z_true': Z_true, 'Y_timing': Y_timing, 'Y_spend': Y_spend,
        'alpha_h_true': alpha_h_true, 'beta_m_true': beta_m_true,
        'r_nb_true': r_nb_true, 'gamma_h_true': gamma_h_true,
        'shape_spend_true': shape_spend_true, 'sparsity': sparsity,
    }
    return dgp

def save_dgp(dgp, outdir="outputs"):
    os.makedirs(outdir, exist_ok=True)
    fname = (f"dgp_N{dgp['N']}_T{dgp['T']}_K{dgp['K']}_gm{dgp['gamma_m']}_"
             f"p0{dgp['pi0']}_ag{dgp['alpha_gamma']}_ss{dgp['state_sep']}_"
             f"st{dgp['sigma_theta']}_regime{dgp['regime']}_seed{dgp['seed']}.pkl")
    fpath = os.path.join(outdir, fname)
    with open(fpath, 'wb') as f:
        pickle.dump(dgp, f)
    print(f"Saved DGP: {fpath}")
    return fpath

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=100)
    parser.add_argument("--T", type=int, default=52)
    parser.add_argument("--K", type=int, default=3)
    parser.add_argument("--gamma_m", type=float, default=0.5)
    parser.add_argument("--pi0", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha_gamma", type=float, default=2.0)
    parser.add_argument("--state_sep", type=str, default="baseline")
    parser.add_argument("--sigma_theta", type=float, default=1.0)
    parser.add_argument("--regime", type=str, default="pure_theta",
                        choices=["pure_theta", "pure_rho", "mixed", "none"])
    parser.add_argument("--outdir", type=str, default="outputs")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    
    dgp = generate_dgp_jrssc(args.N, args.T, args.K, args.gamma_m, args.pi0,
                              args.seed, args.alpha_gamma, args.state_sep,
                              args.sigma_theta, args.regime)
    
    if args.verbose:
        print(f"=== DGP: {dgp['regime']} ===")
        print(f"N={dgp['N']}, T={dgp['T']}, K={dgp['K']}")
        print(f"gamma_m={dgp['gamma_m']}, rho={dgp['rho']}, pi0={dgp['pi0']}")
        print(f"Sparsity: {dgp['sparsity']*100:.1f}%")
    
    save_dgp(dgp, args.outdir)
