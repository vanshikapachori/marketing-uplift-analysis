"""
01_generate_data.py
Simulates a store-level panel dataset for a marketing campaign rollout.

Design:
- 200 stores, observed weekly for 12 weeks (6 weeks pre, 6 weeks post campaign launch)
- Stores are NOT randomly assigned to treatment (realistic: marketing targeted
  larger, urban stores first) -> confounding by store_size and urban flag
- True causal uplift built into the simulation so we can validate our estimate
  against ground truth (this is what makes the project defensible in an interview)
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

N_STORES = 200
N_WEEKS = 12
TREATMENT_WEEK = 6          # campaign launches at week 6 (0-indexed: weeks 6-11 are "post")
TRUE_UPLIFT = 850.0         # ground-truth causal effect: $ extra weekly sales per store

# ---- Store-level attributes (confounders) ----
store_id = np.arange(N_STORES)
store_size = rng.normal(1000, 250, N_STORES).clip(300, None)   # sq. ft (proxy for baseline scale)
urban = rng.binomial(1, 0.45, N_STORES)                         # 1 = urban store

# Non-random targeting: bigger & urban stores are MUCH more likely to be treated
propensity_logit = (
    -3.0
    + 0.0028 * store_size
    + 1.4 * urban
)
propensity = 1 / (1 + np.exp(-propensity_logit))
treated = rng.binomial(1, propensity)

stores = pd.DataFrame({
    "store_id": store_id,
    "store_size": store_size,
    "urban": urban,
    "treated": treated,
})

# ---- Build weekly panel ----
rows = []
for _, s in stores.iterrows():
    baseline = 4000 + 2.8 * s.store_size + 600 * s.urban
    # gentle common time trend (seasonality) shared by all stores
    for week in range(N_WEEKS):
        time_trend = 40 * week
        noise = rng.normal(0, 220)
        post = 1 if week >= TREATMENT_WEEK else 0
        effect = TRUE_UPLIFT * s.treated * post
        sales = baseline + time_trend + effect + noise
        rows.append({
            "store_id": int(s.store_id),
            "week": week,
            "post": post,
            "treated": int(s.treated),
            "store_size": s.store_size,
            "urban": int(s.urban),
            "weekly_sales": round(sales, 2),
        })

panel = pd.DataFrame(rows)
panel.to_csv("/home/claude/uplift_project/campaign_panel.csv", index=False)
print(panel.shape)
print(panel.groupby(["treated", "post"])["weekly_sales"].mean())
print(f"\nGround truth uplift built into simulation: ${TRUE_UPLIFT}/store/week")
