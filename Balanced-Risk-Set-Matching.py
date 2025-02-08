import pandas as pd
import numpy as np
from scipy.spatial import distance
from scipy.stats import wilcoxon
import matplotlib.pyplot as plt

# Generate synthetic data
np.random.seed(24)
n_patients = 400
data = {
    'patient_id': np.arange(0, n_patients),
    'gender': np.random.choice(['M', 'F'], n_patients),
    'treatment_time': np.random.choice([6, 9, 12, np.nan], n_patients),
    'pain_baseline': np.random.randint(0, 10, n_patients),
    'urgency_baseline': np.random.randint(0, 10, n_patients),
    'frequency_baseline': np.random.randint(0, 20, n_patients),
    'pain_treatment': np.random.randint(0, 10, n_patients),
    'urgency_treatment': np.random.randint(0, 10, n_patients),
    'frequency_treatment': np.random.randint(0, 20, n_patients),
    'outcome': np.random.normal(0, 1, n_patients)
}
df = pd.DataFrame(data)

# Split into treated and control
treated = df[df['treatment_time'].notna()].copy()
not_treated = df[df['treatment_time'].isna()].copy()

def mahalanobis_distance(x, y, cov_inv):
    delta = x - y
    return np.sqrt(np.dot(np.dot(delta, cov_inv), delta.T))

# Compute covariance matrix inverse for the covariates
cov_cols = ['pain_baseline', 'urgency_baseline', 'frequency_baseline',
            'pain_treatment', 'urgency_treatment', 'frequency_treatment']
cov_matrix = df[cov_cols].cov().values
cov_inv = np.linalg.inv(cov_matrix)

# Perform greedy matching
matched_pairs = []
used_controls = set()

for idx, treated_patient in treated.iterrows():
    tm = treated_patient['treatment_time']
    eligible_controls = not_treated[~not_treated.index.isin(used_controls)]
    
    # Calculate distances
    distances = []
    for c_idx, control_patient in eligible_controls.iterrows():
        x = treated_patient[cov_cols].values
        y = control_patient[cov_cols].values
        d = mahalanobis_distance(x, y, cov_inv)
        distances.append((c_idx, d))
    
    # Find closest control
    if distances:
        distances.sort(key=lambda x: x[1])
        closest = distances[0]
        matched_pairs.append((treated_patient['patient_id'], closest[0]))
        used_controls.add(closest[0])

# Create matched dataset
matched_treated = treated[treated['patient_id'].isin([p[0] for p in matched_pairs])]
matched_not_treated = not_treated.loc[[p[1] for p in matched_pairs]]

# Check balance before and after matching
def check_balance(col, treated_df, not_treated_df, time):
    plt.figure()
    plt.boxplot([not_treated_df[col], treated_df[col]], labels=['Never/Later Treated', 'Treated'])
    plt.title(f'Balance check for {col} {time}')
    plt.show()

for col in cov_cols:
    check_balance(col, treated, not_treated, "(Before)")
    check_balance(col, matched_treated, matched_not_treated, "(After)")

    # Extract outcomes for matched pairs
treated_outcomes = matched_treated['outcome'].values
not_treated_outcomes = matched_not_treated['outcome'].values

# Perform Wilcoxon test
stat, p_value = wilcoxon(treated_outcomes, not_treated_outcomes)
print(f"Wilcoxon signed-rank test: statistic={stat}, p-value={p_value:.4f}")

def sensitivity_analysis(pairs, gamma_values):
    original_p = wilcoxon(treated_outcomes, not_treated_outcomes).pvalue
    p_values = [original_p]
    for gamma in gamma_values:
        # Adjust p-value bounds based on gamma (simplified)
        upper_p = original_p * gamma
        lower_p = original_p / gamma
        p_values.append((lower_p, upper_p))
    return p_values

gamma_values = [1.5, 2, 3]
sensitivity_results = sensitivity_analysis(matched_pairs, gamma_values)
print("Sensitivity Analysis:")
for gamma, p in zip(gamma_values, sensitivity_results[1:]):
    print(f"Gamma={gamma}: p-range=({p[0]:.4f}, {p[1]:.4f})")