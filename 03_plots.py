import pandas as pd
import matplotlib.pyplot as plt

panel = pd.read_csv("/home/claude/uplift_project/campaign_panel.csv")
results = pd.read_csv("/home/claude/uplift_project/results_summary.csv")

# ---- Plot 1: Weekly sales trend, treated vs control (parallel trends check) ----
trend = panel.groupby(["week", "treated"])["weekly_sales"].mean().unstack()
trend.columns = ["Control", "Treated"]

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(trend.index, trend["Control"], marker="o", label="Control stores", color="#4C72B0")
ax.plot(trend.index, trend["Treated"], marker="o", label="Treated stores", color="#DD8452")
ax.axvline(x=5.5, color="grey", linestyle="--", linewidth=1)
ax.text(5.6, trend.values.max() * 0.98, "Campaign launch", fontsize=9, color="grey")
ax.set_xlabel("Week")
ax.set_ylabel("Avg weekly sales ($)")
ax.set_title("Parallel Trends Check: Treated vs Control Store Sales")
ax.legend()
fig.tight_layout()
fig.savefig("/home/claude/uplift_project/plot_parallel_trends.png", dpi=150)

# ---- Plot 2: Method comparison bar chart with ground truth line ----
fig2, ax2 = plt.subplots(figsize=(7, 5))
colors = ["#C44E52", "#55A868", "#4C72B0"]
bars = ax2.bar(results.method, results.estimate, color=colors)
ax2.axhline(y=850, color="black", linestyle="--", linewidth=1.2, label="Ground truth uplift ($850)")
for bar, val in zip(bars, results.estimate):
    ax2.text(bar.get_x() + bar.get_width() / 2, val + 15, f"${val:,.0f}",
             ha="center", fontsize=10, fontweight="bold")
ax2.set_ylabel("Estimated uplift ($ / store / week)")
ax2.set_title("Naive vs Causal Methods: Recovering the True Campaign Effect")
ax2.set_xticklabels(results.method, rotation=10, ha="right")
ax2.legend()
fig2.tight_layout()
fig2.savefig("/home/claude/uplift_project/plot_method_comparison.png", dpi=150)

print("Saved plot_parallel_trends.png and plot_method_comparison.png")
