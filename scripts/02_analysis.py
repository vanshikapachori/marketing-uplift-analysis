"""
02_analysis.py
Causal uplift analysis of a store-level marketing campaign.

Methods:
  1. Naive post-only comparison (treated vs control) -> shows why this is BIASED
  2. Difference-in-Differences (DiD) via OLS with store-clustered robust SE
  3. Propensity Score Matching (PSM) as a robustness check
  4. Parallel-trends visual check (pre-period only)

No statsmodels available in this environment -> DiD OLS is implemented by hand
with matrix algebra, including a cluster-robust ("clustered by store") variance
estimator, which is the correct SE choice for panel data with repeated
store observations.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

panel = pd.read_csv("/home/claude/uplift_project/campaign_panel.csv")

# ----------------------------------------------------------------------
# 1. NAIVE COMPARISON (the mistake a lot of analysts make)
# ----------------------------------------------------------------------
post = panel[panel.post == 1]
naive_treated_avg = post[post.treated == 1].weekly_sales.mean()
naive_control_avg = post[post.treated == 0].weekly_sales.mean()
naive_diff = naive_treated_avg - naive_control_avg

print("=" * 60)
print("1) NAIVE POST-ONLY COMPARISON (biased - ignore selection)")
print("=" * 60)
print(f"Treated stores avg (post):  ${naive_treated_avg:,.2f}")
print(f"Control stores avg (post):  ${naive_control_avg:,.2f}")
print(f"Naive 'effect':             ${naive_diff:,.2f}  <-- overstated (confounded)")

# ----------------------------------------------------------------------
# 2. DIFFERENCE-IN-DIFFERENCES via manual OLS
#    weekly_sales = b0 + b1*treated + b2*post + b3*(treated*post) + e
#    b3 = the DiD causal estimate
# ----------------------------------------------------------------------
df = panel.copy()
df["treated_post"] = df.treated * df.post

X = np.column_stack([
    np.ones(len(df)),
    df.treated.values,
    df.post.values,
    df.treated_post.values,
])
y = df.weekly_sales.values

XtX_inv = np.linalg.inv(X.T @ X)
beta = XtX_inv @ X.T @ y
resid = y - X @ beta

# Cluster-robust ("clustered by store") sandwich variance estimator
clusters = df.store_id.values
unique_clusters = np.unique(clusters)
meat = np.zeros((X.shape[1], X.shape[1]))
for c in unique_clusters:
    idx = clusters == c
    Xc = X[idx]
    uc = resid[idx].reshape(-1, 1)
    score = Xc.T @ uc
    meat += score @ score.T
n_clusters = len(unique_clusters)
k = X.shape[1]
correction = (n_clusters / (n_clusters - 1)) * ((len(y) - 1) / (len(y) - k))
vcov_cluster = correction * (XtX_inv @ meat @ XtX_inv)
se_cluster = np.sqrt(np.diag(vcov_cluster))

labels = ["intercept", "treated", "post", "treated_x_post (DiD estimate)"]
did_estimate = beta[3]
did_se = se_cluster[3]
ci_low, ci_high = did_estimate - 1.96 * did_se, did_estimate + 1.96 * did_se
t_stat = did_estimate / did_se
# two-sided p-value via normal approx (large N)
from math import erf, sqrt
p_value = 2 * (1 - 0.5 * (1 + erf(abs(t_stat) / sqrt(2))))

print("\n" + "=" * 60)
print("2) DIFFERENCE-IN-DIFFERENCES (OLS, store-clustered SE)")
print("=" * 60)
for lbl, b, se in zip(labels, beta, se_cluster):
    print(f"{lbl:35s} coef = {b:9.2f}   SE = {se:7.2f}")
print(f"\n--> DiD causal uplift estimate: ${did_estimate:,.2f} per store per week")
print(f"    95% CI: [${ci_low:,.2f}, ${ci_high:,.2f}]")
print(f"    p-value: {p_value:.4f}")

# ----------------------------------------------------------------------
# 3. PROPENSITY SCORE MATCHING (robustness check, using pre-period only)
# ----------------------------------------------------------------------
pre_store_level = (
    panel[panel.post == 0]
    .groupby("store_id")
    .agg(store_size=("store_size", "first"),
         urban=("urban", "first"),
         treated=("treated", "first"),
         pre_sales=("weekly_sales", "mean"))
    .reset_index()
)
post_store_level = (
    panel[panel.post == 1]
    .groupby("store_id")
    .agg(post_sales=("weekly_sales", "mean"))
    .reset_index()
)
store_level = pre_store_level.merge(post_store_level, on="store_id")
store_level["individual_growth"] = store_level.post_sales - store_level.pre_sales

X_ps = store_level[["store_size", "urban"]].values
t_ps = store_level["treated"].values

ps_model = LogisticRegression()
ps_model.fit(X_ps, t_ps)
store_level["propensity"] = ps_model.predict_proba(X_ps)[:, 1]

treated_df = store_level[store_level.treated == 1].reset_index(drop=True)
control_df = store_level[store_level.treated == 0].reset_index(drop=True)

nn = NearestNeighbors(n_neighbors=1)
nn.fit(control_df[["propensity"]].values)
dist, idx = nn.kneighbors(treated_df[["propensity"]].values)

matched_control_growth = control_df.iloc[idx.flatten()]["individual_growth"].values
matched_treated_growth = treated_df["individual_growth"].values
psm_effect = matched_treated_growth.mean() - matched_control_growth.mean()
psm_se = np.std(matched_treated_growth - matched_control_growth, ddof=1) / np.sqrt(len(treated_df))

print("\n" + "=" * 60)
print("3) PROPENSITY SCORE MATCHING (1-NN on pre-period covariates)")
print("=" * 60)
print(f"Matched treated stores:  {len(treated_df)}")
print(f"--> PSM uplift estimate: ${psm_effect:,.2f} per store per week")
print(f"    SE (paired):         ${psm_se:,.2f}")

# ----------------------------------------------------------------------
# Save summary results for the README / plotting script
# ----------------------------------------------------------------------
results = pd.DataFrame({
    "method": ["Naive post-only", "Difference-in-Differences", "Propensity Score Matching"],
    "estimate": [naive_diff, did_estimate, psm_effect],
})
results.to_csv("/home/claude/uplift_project/results_summary.csv", index=False)
store_level.to_csv("/home/claude/uplift_project/store_level_matched.csv", index=False)
print("\nSaved: results_summary.csv, store_level_matched.csv")
