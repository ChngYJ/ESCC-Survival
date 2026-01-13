import os
import warnings
warnings.filterwarnings("ignore")

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
from pycox.models import CoxPH
from torchtuples.practical import MLPVanilla
from torchtuples.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer, StandardScaler
from sklearn.compose import ColumnTransformer
from lifelines.utils import concordance_index
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from sklearn.metrics import roc_curve, auc
import optuna

gpu_id = 0
device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

geo_path = "./SR_dataset/GEO_24_clinical_genes.xlsx"
tcga_path = "./SR_dataset/TCGA_24_clinical_genes.xlsx"

GENE_COLS = ['KBTBD12', 'ADGRB3', 'ANGPTL7', 'FAM155B', 'NELL2', 'MAGEA11', 'MAL2',
             'B3GNT3', 'SLCO1B3', 'HPSE', 'HOOK1', 'AMBP', 'POF1B', 'CFHR4', 'PRR9', 'CST1']
CLINIC_NUM = ['Age']
CLINIC_CAT = ['Sex', 'T_stage', 'N_stage', 'Pathologic_stage', 'Tumor_loation']
TARGET_EVENT = 'event'
TARGET_TIME = 'Survival_months'

SEED = 2025
def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
set_seed(SEED)

def load_dataset(path, cohort_name):
    df = pd.read_excel(path)
    df['cohort'] = cohort_name
    return df

geo = load_dataset(geo_path, 'GEO')
tcga = load_dataset(tcga_path, 'TCGA')

need_cols = set(GENE_COLS + CLINIC_NUM + CLINIC_CAT + [TARGET_EVENT, TARGET_TIME, 'cohort'])
for nm, df in [('GEO', geo), ('TCGA', tcga)]:
    miss = need_cols - set(df.columns)
    if miss:
        raise ValueError(f"{nm} 缺少列: {miss}")

df_all = pd.concat([geo, tcga], ignore_index=True)


df_all[TARGET_EVENT] = df_all[TARGET_EVENT].astype(int)
df_all[TARGET_TIME] = pd.to_numeric(df_all[TARGET_TIME], errors='coerce')

DAYS_PER_MONTH = 30.4375
df_all[TARGET_TIME] = df_all[TARGET_TIME] / DAYS_PER_MONTH

df_all = df_all.sample(n=30, random_state=SEED)

TEST_SIZE = 0.2
VAL_SIZE = 0.15
train_val_df, test_df = train_test_split(
    df_all, test_size=TEST_SIZE, random_state=SEED, stratify=df_all[TARGET_EVENT]
)
val_ratio = VAL_SIZE / (1.0 - TEST_SIZE)
train_df, val_df = train_test_split(
    train_val_df, test_size=val_ratio, random_state=SEED, stratify=train_val_df[TARGET_EVENT]
)
print(f"Split sizes -> train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

ALL_CAT = CLINIC_CAT + ['cohort']
nq = int(min(1000, max(50, len(train_df)//2)))
preprocess = ColumnTransformer(
    transformers=[
        ('genes_rankgauss', QuantileTransformer(output_distribution='normal', n_quantiles=nq, random_state=SEED), GENE_COLS),
        ('clin_num', 'passthrough', CLINIC_NUM),
        ('clin_cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ALL_CAT),
    ],
    remainder='drop'
)

def split_xy(df):
    X_raw = df[GENE_COLS + CLINIC_NUM + ALL_CAT].copy()
    y_time = df[TARGET_TIME].astype(float).values
    y_event = df[TARGET_EVENT].astype(int).values
    return X_raw, (y_time, y_event)

Xtr_raw, ytr = split_xy(train_df)
Xva_raw, yva = split_xy(val_df)
Xte_raw, yte = split_xy(test_df)

# fit/transform
Xtr_part = preprocess.fit_transform(Xtr_raw)
Xva_part = preprocess.transform(Xva_raw)
Xte_part = preprocess.transform(Xte_raw)


ohe: OneHotEncoder = preprocess.named_transformers_['clin_cat']
cat_feature_names = ohe.get_feature_names_out(ALL_CAT).tolist()
feature_names = [f"{g}_rankgauss" for g in GENE_COLS] + CLINIC_NUM + cat_feature_names

scaler = StandardScaler()
X_train = scaler.fit_transform(Xtr_part)
X_val   = scaler.transform(Xva_part)
X_test  = scaler.transform(Xte_part)

y_train_time, y_train_event = ytr
y_val_time,   y_val_event   = yva
y_test_time,  y_test_event  = yte

print(f"Final feature dim = {X_train.shape[1]}")

def to_torch_batch(X, t, e):
    X_t = torch.tensor(X, dtype=torch.float32).to(device)
    t_t = torch.tensor(t, dtype=torch.float32).to(device)
    e_t = torch.tensor(e, dtype=torch.int64).to(device)
    return X_t, (t_t, e_t)

X_tr_t, y_tr_t = to_torch_batch(X_train, y_train_time, y_train_event)
X_va_t, y_va_t = to_torch_batch(X_val,   y_val_time,   y_val_event)

def objective(trial):
    hidden_1 = trial.suggest_int("hidden_1", 32, 128)
    hidden_2 = trial.suggest_int("hidden_2", 16, 128)
    dropout = trial.suggest_float("dropout", 0.2, 0.6)
    lr = trial.suggest_float("lr", 5e-4, 5e-3, log=True)
    wd = trial.suggest_float("weight_decay", 1e-5, 5e-4, log=True)

    batch_sz = trial.suggest_categorical("batch_size", [4, 8, 16, 32, 64])
    epochs = 200
    patience = 20

    net = MLPVanilla(X_tr_t.shape[1], [hidden_1, hidden_2], 1,
                     batch_norm=False, dropout=dropout).to(device)
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    model = CoxPH(net, optimizer=optimizer)

    callbacks = [EarlyStopping(patience=patience)]
    model.fit(X_tr_t, y_tr_t, batch_size=batch_sz, epochs=epochs, verbose=False,
              val_data=(X_va_t, y_va_t), callbacks=callbacks)

    with torch.no_grad():
        risk_val = model.predict(X_va_t).reshape(-1).detach().cpu().numpy()
    val_c = concordance_index(y_val_time, -risk_val, y_val_event)
    return val_c

N_TRIALS = 20
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=N_TRIALS)
best_params = study.best_params
print("best_params:", best_params)

H1, H2 = best_params["hidden_1"], best_params["hidden_2"]
DROPOUT = best_params["dropout"]
LR = best_params["lr"]
WEIGHT_DECAY = best_params["weight_decay"]
BATCH_SIZE = best_params["batch_size"]
EPOCHS = 200
PATIENCE = 20

net = MLPVanilla(X_tr_t.shape[1], [H1, H2], 1, batch_norm=False, dropout=DROPOUT).to(device)
optimizer = optim.Adam(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
model = CoxPH(net, optimizer=optimizer)
callbacks = [EarlyStopping(patience=PATIENCE)]
log = model.fit(X_tr_t, y_tr_t, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=True,
                val_data=(X_va_t, y_va_t), callbacks=callbacks)

# C-index（train/val/test）=====
with torch.no_grad():
    risk_train = model.predict(X_tr_t).reshape(-1).detach().cpu().numpy()
    risk_val   = model.predict(X_va_t).reshape(-1).detach().cpu().numpy()

X_te_t, y_te_t = to_torch_batch(X_test, y_test_time, y_test_event)
with torch.no_grad():
    risk_test  = model.predict(X_te_t).reshape(-1).detach().cpu().numpy()

train_c = concordance_index(y_train_time, -risk_train, y_train_event)
val_c   = concordance_index(y_val_time,   -risk_val,   y_val_event)
test_c  = concordance_index(y_test_time,  -risk_test,  y_test_event)
print(f"Train C-index: {train_c:.4f} | Val C-index: {val_c:.4f} | Test C-index: {test_c:.4f}")

model.compute_baseline_hazards()
surv_df = model.predict_surv_df(X_te_t)

def closest_index(target_month, available_times):
    idx = np.argmin(np.abs(available_times - target_month))
    return available_times[idx]

# ROC：12、36、60
time_points = [12, 36, 60]
test_info = test_df[[TARGET_TIME, TARGET_EVENT]].reset_index(drop=True).copy()

plt.figure(figsize=(8, 6), dpi=500)
available_times = surv_df.index.values.astype(float)

for t in time_points:
    t_use = closest_index(t, available_times)
    surv_probs = surv_df.loc[t_use].values
    risk_scores = 1 - surv_probs
    y_true = ((test_info[TARGET_TIME] <= t) & (test_info[TARGET_EVENT] == 1)).astype(int).values
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        print(f"[WARN] t={t} month, skip AUC")
        continue
    fpr, tpr, _ = roc_curve(y_true, risk_scores)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{t} months (AUC = {roc_auc:.2f})")

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
plt.title('DeepSurv Survival Analysis: 1-year, 3-year, and 5-year ROC Curves')
plt.legend(); plt.grid(True)
plt.tight_layout()
plt.show()

# AUC：1~60
t_points = np.arange(1, 61)
auc_values = []
for t in t_points:
    t_use = closest_index(t, available_times)
    surv_probs = surv_df.loc[t_use].values
    risk_scores = 1 - surv_probs
    y_true = ((test_info[TARGET_TIME] <= t) & (test_info[TARGET_EVENT] == 1)).astype(int).values
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        auc_values.append(np.nan)
    else:
        fpr, tpr, _ = roc_curve(y_true, risk_scores)
        auc_values.append(auc(fpr, tpr))

plt.figure(figsize=(8, 6), dpi=500)
plt.plot(t_points, auc_values, marker='o')
plt.xlabel("Time (months)"); plt.ylabel("AUC")
plt.title('DeepSurv Survival Analysis: AUC Over Time')
plt.grid(True); plt.ylim([0.5, 1.0])
plt.tight_layout()
plt.show()

# save
os.makedirs("./Log", exist_ok=True)
df_auc = pd.DataFrame({"Month": t_points, "DeepSurv_AUC": auc_values})
df_auc.to_excel("./Log/DeepSurv_AUC.xlsx", index=False)
print("DeepSurv_AUC.xlsx save！")

# K-M + Log-rank
X_all_raw = df_all[GENE_COLS + CLINIC_NUM + ALL_CAT].copy()
X_all_part = preprocess.transform(X_all_raw)
X_all = scaler.transform(X_all_part)

X_all_t = torch.tensor(X_all, dtype=torch.float32).to(device)
with torch.no_grad():
    risk_scores_all = model.predict(X_all_t).reshape(-1).detach().cpu().numpy()

df_all_km = df_all.copy()
df_all_km['risk_score'] = risk_scores_all

median_cut = df_all_km['risk_score'].median()
print(f"[Risk Split] median threshold (months-based model) = {median_cut:.6f}")
df_all_km['risk_group'] = np.where(df_all_km['risk_score'] <= median_cut, 'Low', 'High')

out_xlsx = "./log/deepsurv_risk_groups.xlsx"
df_all_km.to_excel(out_xlsx, index=False)
print(f"risk grouping saved: {out_xlsx}")

# K-M
df_km = df_all_km.dropna(subset=[TARGET_TIME, TARGET_EVENT, 'risk_group']).copy()
kmf = KaplanMeierFitter()

plt.figure(figsize=(10, 7), dpi=600)
ax = plt.gca()

colors = {'Low': 'green', 'High': 'red'}
for group in ['Low', 'High']:
    mask = df_km['risk_group'] == group
    if mask.sum() == 0:
        continue
    kmf.fit(df_km.loc[mask, TARGET_TIME], df_km.loc[mask, TARGET_EVENT], label=group)
    kmf.plot_survival_function(ax=ax, ci_show=True, color=colors[group], linestyle='-', linewidth=2.5)

plt.title("Kaplan–Meier Survival Curves (High vs Low Risk) — Time in Months", fontsize=16)
plt.xlabel("Time (months)", fontsize=13)
plt.ylabel("Survival Probability", fontsize=13)
plt.xticks(np.arange(0, 61, 12), fontsize=12)
plt.yticks(np.linspace(0, 1, 11), fontsize=12)
plt.xlim(0, 60)
plt.ylim(0, 1.01)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(title="Risk Group", fontsize=11, title_fontsize=12, loc='best', frameon=True)
plt.tight_layout()
km_path = "./log/KM_Curve_By_Risk_Group.jpg"
plt.savefig(km_path, bbox_inches='tight', dpi=600)
plt.show()

# Log-rank
low_grp = df_km[df_km['risk_group'] == 'Low']
high_grp = df_km[df_km['risk_group'] == 'High']

if len(low_grp) > 0 and len(high_grp) > 0:
    try:
        lr_res = logrank_test(
            low_grp[TARGET_TIME], high_grp[TARGET_TIME],
            event_observed_A=low_grp[TARGET_EVENT],
            event_observed_B=high_grp[TARGET_EVENT]
        )
        print(f"Log-rank（Low vs High）: p-value = {lr_res.p_value:.4e}, test_statistic = {lr_res.test_statistic:.3f}")
    except Exception as e:
        print(f"Log-rank error：{e}")
else:
    print("The risk group sample is insufficient to perform Log rank testing.")
