import pandas as pd
from scipy.stats import ttest_1samp, ttest_ind, sem, t

# Load data
df = pd.read_csv("/Users/icarus/Desktop/event_driven/monetary_policy/index_30d_returns.csv")

# Clean and parse
df['FY'] = df['FY'].astype(str)
df = df[df['Return_30d'].notna()]
df = df[df['Return_30d'] != 0.0]

# Split by FY
pre_df = df[df['FY'] <= '2016-17']
post_df = df[df['FY'] >= '2017-18']

# Function to compute confidence interval
def compute_confidence_interval(data, confidence=0.90):
    n = len(data)
    if n < 2:
        return (None, None)
    m = data.mean()
    se = sem(data)
    h = se * t.ppf((1 + confidence) / 2., n - 1)
    return m - h, m + h

# One-sample t-test per sector
def run_ttest(df_period, label):
    print(f"\n=== {label} ===")
    print(f"{'Sector':<15} {'Mean':<10} {'p-value':<10} {'Significant':<12} {'95% CI':<25} {'N':<5}")
    print("-" * 80)
    
    for sector in sorted(df_period['IndexName'].unique()):
        sector_returns = df_period[df_period['IndexName'] == sector]['Return_30d']
        if len(sector_returns) < 5:
            continue
        mean_return = sector_returns.mean()
        t_stat, p_value = ttest_1samp(sector_returns, 0)
        ci_low, ci_high = compute_confidence_interval(sector_returns)
        significant = "YES" if p_value < 0.1 else "NO"
        ci_str = f"[{ci_low:.4f}, {ci_high:.4f}]" if ci_low is not None else "N/A"
        print(f"{sector:<15} {mean_return:>8.4f}   {p_value:>8.4f}   {significant:<12} {ci_str:<25} {len(sector_returns):<5}")

# Run for each group
run_ttest(pre_df, "Pre-FY2017 Policies")
run_ttest(post_df, "Post-FY2017 Policies")

# Compare mean return differences
print("\n=== Change in Mean Returns: Post vs Pre FY2017 ===")
print(f"{'Sector':<15} {'Δ Mean':<10} {'p-value':<10} {'Significant':<12} {'N_Pre':<6} {'N_Post':<6}")
print("-" * 75)

for sector in sorted(df['IndexName'].unique()):
    r1 = pre_df[pre_df['IndexName'] == sector]['Return_30d']
    r2 = post_df[post_df['IndexName'] == sector]['Return_30d']
    if len(r1) >= 5 and len(r2) >= 5:
        stat, p = ttest_ind(r1, r2, equal_var=False)
        delta_mean = r2.mean() - r1.mean()
        significant = "YES" if p < 0.1 else "NO"
        print(f"{sector:<15} {delta_mean:>8.4f}   {p:>8.4f}   {significant:<12} {len(r1):<6} {len(r2):<6}")
