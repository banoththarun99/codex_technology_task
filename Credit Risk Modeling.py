import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss, roc_curve
import lightgbm as lgb
import matplotlib.pyplot as plt
import os

# ---------------- PARAMETERS ----------------
data_path = r"C:\Users\Banoth Sudhar\Desktop\Tharun\DS project\data.csv"
output_dir = r"C:\Users\Banoth Sudhar\Desktop\Tharun\DS project\output"
os.makedirs(output_dir, exist_ok=True)
n_splits = 5  # K-Fold

# ---------------- LOAD OR SIMULATE DATA ----------------
try:
    data = pd.read_csv(data_path)
    print(f"Loaded {data.shape[0]} rows from {data_path}")
except:
    print("Data file not found, generating synthetic dataset...")
    n = 1000
    np.random.seed(42)
    data = pd.DataFrame({
        'age': np.random.randint(20, 65, n),
        'income': np.random.randint(20000, 150000, n),
        'loan_amount': np.random.randint(5000, 50000, n),
        'credit_score': np.random.randint(300, 850, n),
        'previous_defaults': np.random.randint(0, 3, n),
        'employment_years': np.random.randint(0, 40, n),
    })
    prob_default = 1 / (1 + np.exp(-(0.00005*data['loan_amount'] - 0.00003*data['income'] 
                                    - 0.005*data['credit_score'] + 0.5*data['previous_defaults'])))
    data['default'] = np.random.binomial(1, prob_default)
    print(f"Synthetic dataset generated: {data.shape[0]} rows")

# ---------------- FEATURE ENGINEERING ----------------
data['credit_utilization'] = data['loan_amount'] / (data['income'] + 1e-5)
data['debt_to_income'] = data['loan_amount'] / (data['income'] + 1e-5)
data['installment_to_income'] = (data['loan_amount']/12) / (data['income'] + 1e-5)
data['num_total_dels'] = data['previous_defaults']
data['account_age_days'] = data['employment_years'] * 365
data['days_since_last_delinq'] = 365  # placeholder
features = ['credit_utilization', 'debt_to_income', 'installment_to_income',
            'num_total_dels', 'account_age_days', 'days_since_last_delinq', 'loan_amount']

X = data[features].copy()
y = data['default']

# ---------------- K-FOLD CROSS-VALIDATION ----------------
logistic_preds = np.zeros(len(data))
lgbm_preds = np.zeros(len(data))
logistic_metrics = []
lgbm_metrics = []

kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

for train_idx, test_idx in kf.split(X):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # Logistic Regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    log_model = LogisticRegression(max_iter=1000)
    log_model.fit(X_train_scaled, y_train)
    y_pred_log = log_model.predict_proba(X_test_scaled)[:, 1]
    logistic_preds[test_idx] = y_pred_log

    auc_log = roc_auc_score(y_test, y_pred_log)
    brier_log = brier_score_loss(y_test, y_pred_log)
    fpr, tpr, _ = roc_curve(y_test, y_pred_log)
    ks_log = max(tpr - fpr)
    logistic_metrics.append({'auc': auc_log, 'brier': brier_log, 'ks': ks_log})

    # LightGBM
    if len(data) >= 50:
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_valid = lgb.Dataset(X_test, y_test, reference=lgb_train)
        lgb_params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'learning_rate': 0.05,
            'num_leaves': 31,
            'seed': 42
        }
        lgb_model = lgb.train(
            lgb_params, lgb_train,
            valid_sets=[lgb_train, lgb_valid],
            num_boost_round=200,
            early_stopping_rounds=20,
            verbose_eval=False
        )
        y_pred_lgb = lgb_model.predict(X_test)
        lgbm_preds[test_idx] = y_pred_lgb

        auc_lgb = roc_auc_score(y_test, y_pred_lgb)
        brier_lgb = brier_score_loss(y_test, y_pred_lgb)
        fpr, tpr, _ = roc_curve(y_test, y_pred_lgb)
        ks_lgb = max(tpr - fpr)
        lgbm_metrics.append({'auc': auc_lgb, 'brier': brier_lgb, 'ks': ks_lgb})
    else:
        lgbm_preds[test_idx] = 0.0
        lgbm_metrics.append({'auc': 0.5, 'brier': 0.25, 'ks': 0.0})

# Aggregate metrics
def mean_metrics(metrics_list):
    return {
        'auc': np.mean([m['auc'] for m in metrics_list]),
        'brier': np.mean([m['brier'] for m in metrics_list]),
        'ks': np.mean([m['ks'] for m in metrics_list])
    }

results_summary = {
    'logistic': mean_metrics(logistic_metrics),
    'lightgbm': mean_metrics(lgbm_metrics)
}

data['logistic_prob'] = logistic_preds
data['lgbm_prob'] = lgbm_preds

print("Cross-validated metrics summary:", results_summary)

# ---------------- DYNAMIC DECILES ----------------
def create_deciles(data, prob_col, decile_col):
    n_unique = data[prob_col].nunique()
    n_bins = min(10, n_unique)
    if n_bins > 1:
        data[decile_col] = pd.qcut(data[prob_col], n_bins, labels=False) + 1
    else:
        data[decile_col] = 1
    return data

data = create_deciles(data, 'logistic_prob', 'logistic_decile')
data = create_deciles(data, 'lgbm_prob', 'lgbm_decile')

# ---------------- DECILE REPORT ----------------
def decile_report(data, decile_col='logistic_decile', target='default', output_file=None):
    report = data.groupby(decile_col)[target].agg(['count','sum']).rename(columns={'sum':'defaults'})
    report = report.sort_index(ascending=False)
    if output_file:
        report.to_csv(output_file)
    return report

logistic_decile_report = decile_report(
    data, decile_col='logistic_decile', target='default',
    output_file=os.path.join(output_dir, 'logistic_decile_report.csv')
)

lgbm_decile_report = decile_report(
    data, decile_col='lgbm_decile', target='default',
    output_file=os.path.join(output_dir, 'lgbm_decile_report.csv')
)

# ---------------- LIFT CHART ----------------
def plot_lift_from_decile(report, model_name='Model'):
    lift_rates = (report['defaults'] / report['count']).sort_index(ascending=False)
    plt.figure(figsize=(6,4))
    plt.bar(range(1, len(lift_rates)+1), lift_rates, color='skyblue')
    plt.xlabel('Decile (1 = highest risk)')
    plt.ylabel('Default Rate')
    plt.title(f'Decile Lift Chart - {model_name}')
    plt.xticks(range(1, len(lift_rates)+1))
    plt.grid(axis='y')
    plt.show()

plot_lift_from_decile(logistic_decile_report, 'Logistic')
plot_lift_from_decile(lgbm_decile_report, 'LightGBM')

# ---------------- KS STATISTICS ----------------
def ks_stat(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return max(tpr - fpr)

print(f"KS Logistic: {ks_stat(data['default'], data['logistic_prob']):.4f}")
print(f"KS LightGBM: {ks_stat(data['default'], data['lgbm_prob']):.4f}")

# ---------------- CREDIT SCORECARD ----------------
def prob_to_score(prob, base_score=600, pdo=20, base_odds=50):
    factor = pdo / np.log(2)
    offset = base_score - factor * np.log(base_odds)
    return offset + factor * np.log((1 - prob) / prob)

data['credit_score'] = data['logistic_prob'].apply(prob_to_score).round(0)

def assign_risk_tier(score):
    if score >= 750: return 'Excellent'
    elif score >= 700: return 'Good'
    elif score >= 650: return 'Fair'
    elif score >= 600: return 'Average'
    elif score >= 550: return 'Below Average'
    else: return 'Poor'

data['risk_tier'] = data['credit_score'].apply(assign_risk_tier)

# ---------------- SAVE ALL OUTPUTS ----------------
data.to_csv(os.path.join(output_dir, 'predictions_with_scorecard_and_tiers.csv'), index=False)
print("All outputs saved successfully, including deciles, scorecard, and risk tiers.")

# ---------------- PLOTS ----------------
plt.figure(figsize=(6,4))
plt.hist(data['credit_score'], bins=20, color='skyblue', edgecolor='black')
plt.xlabel('Credit Score')
plt.ylabel('Number of Borrowers')
plt.title('Credit Score Distribution')
plt.grid(axis='y')
plt.show()

plt.figure(figsize=(6,4))
data['risk_tier'].value_counts().sort_index().plot(kind='bar', color='skyblue', edgecolor='black')
plt.xlabel('Risk Tier')
plt.ylabel('Number of Borrowers')
plt.title('Risk Tier Distribution')
plt.grid(axis='y')
plt.show()
