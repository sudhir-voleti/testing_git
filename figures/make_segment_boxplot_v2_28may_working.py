#!/usr/bin/env python3
"""
Generate boxplot of segment CLV distributions - v2 with formatting fixes.
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

def get_segment_clvs(pkl_path, n_clusters=3):
    """Extract segment CLVs from a PKL."""
    with open(pkl_path, 'rb') as f:
        fit = pickle.load(f)
    
    post = fit['idata'].posterior
    with open(fit['dgp_path'], 'rb') as f:
        dgp = pickle.load(f)
    
    N = dgp['N']
    T = dgp['T']
    
    # Extract theta
    if 'theta' in post:
        theta = post['theta'].mean(dim=['chain','draw']).values.squeeze()
        het_name = 'theta'
    elif 'u' in post and 'v' in post:
        if 'rho' in post:
            u = post['u'].mean(dim=['chain','draw']).values.squeeze()
            v = post['v'].mean(dim=['chain','draw']).values.squeeze()
            theta = np.column_stack([u, v])
            het_name = 'u_v_2d'
        else:
            theta = post['v'].mean(dim=['chain','draw']).values.squeeze()
            het_name = 'v'
    else:
        return None, None, None
    
    theta = np.asarray(theta)
    if theta.ndim == 0:
        theta = np.full(N, theta.item())
    elif theta.ndim == 1 and theta.size == 1:
        theta = np.full(N, theta[0])
    elif theta.ndim == 1:
        pass
    else:
        theta = theta.reshape(-1) if theta.shape[1] == 1 else theta
    
    # K-means
    X = theta.reshape(-1, 1) if theta.ndim == 1 else theta
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    
    # Compute CLV
    alpha_h = post['alpha_h'].mean(dim=['chain','draw']).values
    beta_m = post['beta_m'].mean(dim=['chain','draw']).values
    gamma_m = float(post['gamma_m'].mean(dim=['chain','draw']).values) if 'gamma_m' in post else 0.0
    
    delta_weekly = (1.10)**(1/52) - 1
    clv = np.zeros(N)
    for i in range(N):
        z = int(dgp['Z_true'][i, T-1])
        for t in range(104):
            if het_name == 'theta':
                lam = np.exp(alpha_h[z] + theta[i])
                mu = np.exp(beta_m[z] + gamma_m * theta[i])
            elif het_name == 'u_v_2d':
                lam = np.exp(alpha_h[z] + theta[i, 0])
                mu = np.exp(beta_m[z] + gamma_m * theta[i, 1])
            else:
                lam = np.exp(alpha_h[z])
                mu = np.exp(beta_m[z] + gamma_m * theta[i])
            clv[i] += (lam * mu) / ((1 + delta_weekly)**t)
            z = np.random.choice(len(alpha_h), p=post['Gamma'].mean(dim=['chain','draw']).values[z, :])
    
    # Sort segments by CLV
    seg_clvs = []
    for seg in range(n_clusters):
        mask = labels == seg
        seg_clvs.append(clv[mask])
    
    order = np.argsort([np.mean(c) for c in seg_clvs])[::-1]
    seg_clvs = [seg_clvs[i] for i in order]
    
    return seg_clvs, labels, het_name

def main():
    # Use N=200 PKLs from N250_tests folder
    models = {
        'BEMMAOR-indiv': 'N250_tests/structural/fit_BEMMAOR-indiv-rfree_structural_N200_T52_seed3000.pkl',
        'Heckman-indiv': 'N250_tests/structural/fit_Heckman-rfree_structural_N200_T52_seed3000.pkl',
        'Hurdle-indiv': 'N250_tests/structural/fit_Hurdle-rfree_structural_N200_T52_seed3000.pkl',
    }
    
    fallback_models = {
        'BEMMAOR-indiv': 'structural/rep00/fit_BEMMAOR-indiv-rfree_structural_N50_T52.pkl',
        'Heckman-indiv': 'structural/rep00/fit_Heckman-rfree_structural_N50_T52.pkl',
        'Hurdle-indiv': 'structural/rep00/fit_Hurdle-rfree_structural_N50_T52.pkl',
    }
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, (model_name, pkl_path) in enumerate(models.items()):
        if not Path(pkl_path).exists():
            pkl_path = fallback_models[model_name]
        
        seg_clvs, labels, het_name = get_segment_clvs(pkl_path)
        if seg_clvs is None:
            continue
        
        # Custom boxplot with black median and solid outliers
        bp = axes[idx].boxplot(
            seg_clvs, 
            labels=['High', 'Med', 'Low'],
            patch_artist=True,
            medianprops=dict(color='black', linewidth=2),
            whiskerprops=dict(color='gray', linewidth=1.5),
            capprops=dict(color='gray', linewidth=1.5),
            flierprops=dict(marker='o', markerfacecolor='black', markeredgecolor='black', markersize=4, alpha=0.6)
        )
        
        colors = ['#d62728', '#ff7f0e', '#2ca02c']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.5)
        
        axes[idx].set_title(model_name, fontsize=12, fontweight='bold')
        axes[idx].set_ylabel('CLV ($)', fontsize=10)
        axes[idx].set_xlabel('Segment', fontsize=10)
        axes[idx].ticklabel_format(style='plain', axis='y')
        
        # Place mu values just below ceiling
        y_max = max([max(c) for c in seg_clvs])
        y_min = min([min(c) for c in seg_clvs])
        y_range = y_max - y_min
        text_y = y_max - 0.05 * y_range  # 5% below ceiling
        
        for i, clvs in enumerate(seg_clvs):
            mean_clv = np.mean(clvs)
            axes[idx].text(i+1, text_y, f'μ={mean_clv:.0f}', 
                          ha='center', va='top', fontsize=9, fontweight='bold', color='black')
    
    plt.suptitle('Segment CLV Distributions: Structural World', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('figures/segment_clv_boxplots_v2.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('figures/segment_clv_boxplots_v2.png', dpi=300, bbox_inches='tight')
    print("Saved: figures/segment_clv_boxplots_v2.pdf and .png")

if __name__ == '__main__':
    main()
