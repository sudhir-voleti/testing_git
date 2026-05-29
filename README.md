# BEMMAOR Simulation Study

## Structure
- `dgp/` - Data generating processes
- `models/` - Estimation models (BEMMAOR, Heckman, Hurdle)
- `extractors/` - Post-processing scripts
- `figures/` - Visualization scripts
- `results/` - Output CSVs

## Usage
1. Generate DGPs: `python dgp/dgp_rfm_28may_working.py --world structural --N 200`
2. Run models: `python models/bemmaor_indiv.py --dgp_path path/to/dgp.pkl`
3. Extract metrics: `python extractors/extract_phase1_28may_working.py --base_dir .`

## Key Files
- `dgp_rfm_28may_working.py` - DGP with T=52 + T+40 OOS, RFM features
- `bemmaor_indiv.py` - BEMMAOR-indiv with theta_i heterogeneity
- `extract_phase3_28may_working_v2.py` - Full BDT metrics (CLV, VoI, targeting lift)

## Results
- 180 PKLs extracted across 4 worlds x 6 models
- BEMMAOR-indiv dominates on ARI, whale F1, targeting lift
- Phase 4 segmentation: 3.9x CLV discrimination for BEMMAOR-indiv

## Citation
[Placeholder for paper citation]
