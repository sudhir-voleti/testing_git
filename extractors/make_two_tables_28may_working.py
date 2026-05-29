#!/usr/bin/env python3
"""
Generate two Model x World tables as CSV previews.
"""
import pandas as pd
import numpy as np

df1 = pd.read_csv('overnight_28may/phase1_overnight_180.csv')
df3 = pd.read_csv('overnight_28may/phase3_enhanced_overnight_180.csv')

df = df3.merge(df1[['pkl_path', 'ari_train', 'ari_holdout']], on='pkl_path', how='left')
df['model_name'] = df['model_name'].replace('heckman-indiv-rfree', 'Heckman-indiv-rfree')

models = ['BEMMAOR-indiv-rfree', 'Heckman-indiv-rfree', 'Heckman-global-rfree', 'Hurdle-rfree', 'BEMMAOR-global-rfree']
worlds = ['structural', 'independent', 'correlated', 'mixed']

def format_cell_mean_std(subset, metrics):
    """Format cell with mean ± std for multiple metrics stacked."""
    lines = []
    for metric in metrics:
        mean = subset[metric].mean()
        std = subset[metric].std()
        if np.isnan(mean):
            lines.append(metric + ': —')
        else:
            if metric in ['ari_train', 'whale_f1']:
                lines.append(metric + ': {:.3f}±{:.3f}'.format(mean, std))
            elif metric in ['pp_clv_pred_mean']:
                lines.append(metric + ': {:.0f}±{:.0f}'.format(mean, std))
            elif metric in ['targeting_lift']:
                lines.append(metric + ': {:.1f}%±{:.1f}'.format(mean, std))
            elif metric in ['lead_time_train_mean']:
                lines.append(metric + ': {:.2f}±{:.2f}'.format(mean, std))
            else:
                lines.append(metric + ': {:.2f}±{:.2f}'.format(mean, std))
    return '\n'.join(lines)

# Table 1: Business Metrics
print("=" * 80)
print("TABLE 1: BUSINESS METRICS (Model Quality)")
print("=" * 80)
print("Metrics: ARI, pp-CLV, Whale F1")
print()

table1_data = []
for model in models:
    row = {'Model': model}
    for world in worlds:
        subset = df[(df['model_name'] == model) & (df['world'] == world)]
        if len(subset) > 0:
            row[world] = format_cell_mean_std(subset, ['ari_train', 'pp_clv_pred_mean', 'whale_f1'])
        else:
            row[world] = '—'
    table1_data.append(row)

table1_df = pd.DataFrame(table1_data)
print(table1_df.to_string(index=False))
table1_df.to_csv('overnight_28may/table1_business_metrics.csv', index=False)
print("\nSaved: overnight_28may/table1_business_metrics.csv")

# Table 2: Business Outcomes
print("\n" + "=" * 80)
print("TABLE 2: BUSINESS OUTCOMES (Managerial Value)")
print("=" * 80)
print("Metrics: Targeting Lift, Lead Time, Timing Accuracy")
print()

table2_data = []
for model in models:
    row = {'Model': model}
    for world in worlds:
        subset = df[(df['model_name'] == model) & (df['world'] == world)]
        if len(subset) > 0:
            row[world] = format_cell_mean_std(subset, ['targeting_lift', 'lead_time_train_mean', 'timing_accuracy'])
        else:
            row[world] = '—'
    table2_data.append(row)

table2_df = pd.DataFrame(table2_data)
print(table2_df.to_string(index=False))
table2_df.to_csv('overnight_28may/table2_business_outcomes.csv', index=False)
print("\nSaved: overnight_28may/table2_business_outcomes.csv")

# Also save wide format for LaTeX
print("\n" + "=" * 80)
print("WIDE FORMAT FOR LATEX")
print("=" * 80)

wide_rows = []
for model in models:
    for world in worlds:
        subset = df[(df['model_name'] == model) & (df['world'] == world)]
        if len(subset) > 0:
            wide_rows.append({
                'Model': model,
                'World': world,
                'ARI_mean': subset['ari_train'].mean(),
                'ARI_std': subset['ari_train'].std(),
                'ppCLV_mean': subset['pp_clv_pred_mean'].mean(),
                'ppCLV_std': subset['pp_clv_pred_mean'].std(),
                'WhaleF1_mean': subset['whale_f1'].mean(),
                'WhaleF1_std': subset['whale_f1'].std(),
                'Lift_mean': subset['targeting_lift'].mean(),
                'Lift_std': subset['targeting_lift'].std(),
                'LeadTime_mean': subset['lead_time_train_mean'].mean(),
                'LeadTime_std': subset['lead_time_train_mean'].std(),
                'TimingAcc_mean': subset['timing_accuracy'].mean(),
                'TimingAcc_std': subset['timing_accuracy'].std(),
            })

wide_df = pd.DataFrame(wide_rows)
wide_df.to_csv('overnight_28may/table_wide_for_latex.csv', index=False)
print("Saved: overnight_28may/table_wide_for_latex.csv")
