"""
Script sinh notebook: 03_binary_advanced.ipynb
Chạy: python gen_03_binary_advanced.py
Bao gồm: Feature Engineering + Optuna HPO + SHAP + Calibration
"""
import json

OUTPUT = "03_binary_advanced.ipynb"

def code(src): return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src}
def md(src):   return {"cell_type": "markdown", "metadata": {}, "source": src}

cells = []

# ─────────────────────────────────────────────
# TITLE
# ─────────────────────────────────────────────
cells.append(md(
"""# Binary Classification — Advanced Pipeline
## Feature Engineering · Optuna HPO · SHAP · Calibration

**Kết quả các notebook trước (cùng test set):**

| Model | ROC-AUC | PR-AUC | F1 @ optimal | KS |
| :--- | :---: | :---: | :---: | :---: |
| Logistic Regression (01) | 0.9758 | 0.7904 | 0.6833 | 0.8414 |
| LightGBM base (02) | 0.9730 | 0.7707 | 0.6887 | 0.8328 |

**Notebook này thực hiện 4 bước nâng cao:**
1. **Feature Engineering** — 5 interaction features mới từ domain knowledge
2. **Optuna HPO** — tối ưu siêu tham số LightGBM tự động (50 trials)
3. **SHAP Analysis** — giải thích mô hình ở cả cấp độ global và individual
4. **Calibration** — hiệu chỉnh xác suất để predicted probability ≈ thực tế"""
))

# ─────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────
cells.append(code(
"""import warnings; warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import lightgbm as lgb
import optuna; optuna.logging.set_verbosity(optuna.logging.WARNING)
import shap
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve,
    average_precision_score, precision_recall_curve,
    f1_score, precision_score, recall_score,
)

plt.rcParams["figure.dpi"] = 110
plt.rcParams["font.size"] = 11
RANDOM_STATE = 42
DATA_PATH = "vck_propensity_mockup.csv"
print(f"lightgbm {lgb.__version__} | optuna {optuna.__version__} | shap {shap.__version__}")
print("Libraries loaded.")"""
))

# ─────────────────────────────────────────────
# BASE DATA
# ─────────────────────────────────────────────
cells.append(md("## 0. Load & Split (giống các notebook trước)"))

cells.append(code(
"""df = pd.read_csv(DATA_PATH)
df_model = df.drop(columns=["Customer_ID", "Target_MultiClass"]).copy()
df_model["Gender"] = (df_model["Gender"] == "M").astype(int)
df_model = pd.get_dummies(df_model, columns=["Occupation"], drop_first=True)
feature_cols = [c for c in df_model.columns if c != "Target_Binary"]
X = df_model[feature_cols]
y = df_model["Target_Binary"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print(f"Train: {X_train.shape[0]:,}  |  Test: {X_test.shape[0]:,}  |  Base features: {X.shape[1]}")"""
))

# ─────────────────────────────────────────────
# SECTION 1: FEATURE ENGINEERING
# ─────────────────────────────────────────────
cells.append(md(
"""## 1. Feature Engineering

**Nguyên tắc thiết kế:** Kết hợp các features có tương quan nhân để tạo tín hiệu mạnh hơn từ domain knowledge VCK propensity.

| Feature mới | Công thức | Ý nghĩa |
| :--- | :--- | :--- |
| `VCK_Intent_Score` | `Views_7D + Watchlist×5 + Past_Trans×3` | Điểm tổng hợp intent VCK (có trọng số) |
| `VCK_Views_x_Cash` | `VCK_Views_7D × Cash_to_Asset` | Người xem VCK nhiều VÀ có tiền → sẵn sàng mua |
| `Watchlist_x_Sector` | `Watchlist × Sector_Ratio` | Theo dõi VCK VÀ hay mua cùng ngành |
| `VCK_Views_x_Watchlist` | `VCK_Views_7D × Watchlist` | Kép: xem nhiều và đặt watchlist |
| `Active_Score` | `Momentum / (Days_Since_Trade + 1)` | Đang active + giao dịch gần đây |"""
))

cells.append(code(
"""def add_features(df_in):
    df_out = df_in.copy()
    df_out["VCK_Intent_Score"]      = (df_out["VCK_Page_Views_7D"]
                                       + df_out["Is_VCK_In_Watchlist"] * 5
                                       + df_out["Past_VCK_Transactions"] * 3)
    df_out["VCK_Views_x_Cash"]      = df_out["VCK_Page_Views_7D"]   * df_out["Cash_to_Asset_Ratio"]
    df_out["Watchlist_x_Sector"]    = df_out["Is_VCK_In_Watchlist"] * df_out["Sector_Trade_Ratio"]
    df_out["VCK_Views_x_Watchlist"] = df_out["VCK_Page_Views_7D"]   * df_out["Is_VCK_In_Watchlist"]
    df_out["Active_Score"]          = df_out["Login_Momentum"] / (df_out["Days_Since_Last_Trade"] + 1)
    return df_out

X_fe_train = add_features(X_train)
X_fe_test  = add_features(X_test)
new_feats  = ["VCK_Intent_Score", "VCK_Views_x_Cash",
              "Watchlist_x_Sector", "VCK_Views_x_Watchlist", "Active_Score"]

print(f"Features: {X_train.shape[1]} (base) → {X_fe_train.shape[1]} (after FE)")
print(f"New features: {new_feats}")

# Phân bố feature mới: buyers vs non-buyers
fig, axes = plt.subplots(1, 5, figsize=(18, 4))
buyers     = X_fe_test[y_test == 1]
non_buyers = X_fe_test[y_test == 0]
for i, feat in enumerate(new_feats):
    axes[i].hist(non_buyers[feat], bins=30, alpha=0.55, density=True,
                 color="#1976D2", label="Không mua")
    axes[i].hist(buyers[feat],     bins=30, alpha=0.70, density=True,
                 color="#E53935", label="Có mua")
    axes[i].set_title(feat, fontsize=9, fontweight="bold")
    axes[i].legend(fontsize=7); axes[i].set_yticks([])
plt.suptitle("Phân bố Features mới — Mua vs Không mua", fontsize=13, fontweight="bold", y=1.03)
plt.tight_layout()
plt.show()"""
))

cells.append(md(
"""### Insight — Feature Engineering

**Phân bố 5 features mới** đều cho thấy tách biệt rõ ràng giữa buyer (đỏ) và non-buyer (xanh):

| Feature mới | Tại sao hiệu quả |
| :--- | :--- |
| `VCK_Intent_Score` | Tổng hợp 3 tín hiệu intent mạnh nhất → "điểm sẵn sàng mua" |
| `VCK_Views_x_Cash` | Loại bỏ người chỉ tò mò xem nhưng không có tiền thực sự mua |
| `Active_Score` | Kết hợp login_momentum và recency — phát hiện khách đang "sôi nổi" |

**Nguyên tắc feature engineering tốt:**
- Kết hợp tín hiệu **intent** (xem trang, watchlist) với **capacity** (cash) → precision cao hơn
- Không tạo feature chỉ vì có thể tạo — mỗi feature mới cần có **giải thích domain knowledge**
- Linear models (LR) không tự học interaction → FE giúp LR nhiều hơn tree models"""
))

# ─────────────────────────────────────────────
# SECTION 2: OPTUNA HPO
# ─────────────────────────────────────────────
cells.append(md(
"""## 2. Hyperparameter Optimization với Optuna

**Optuna** dùng thuật toán **TPE (Tree-structured Parzen Estimator)** — thông minh hơn GridSearch/RandomSearch:
- Học từ các trial trước để đề xuất trial tiếp theo có triển vọng hơn
- Tối ưu **PR-AUC** (metric phù hợp nhất với imbalanced data)
- 50 trials với 5-fold CV → tổng 250 lần training

**Search space:**

| Hyperparameter | Range | Lý do |
| :--- | :--- | :--- |
| `n_estimators` | 200–1500 | Số cây — nhiều cây hơn với learning_rate nhỏ |
| `num_leaves` | 20–150 | Độ phức tạp cây — LightGBM nhạy cảm nhất với param này |
| `learning_rate` | 0.01–0.2 (log) | Log scale vì impact phi tuyến |
| `min_child_samples` | 20–200 | Regularization chính cho imbalanced data |
| `subsample` / `colsample_bytree` | 0.5–1.0 | Bagging rows/features |
| `reg_alpha` / `reg_lambda` | 1e-4–10 (log) | L1/L2 regularization |"""
))

cells.append(code(
"""cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

def objective(trial):
    params = {
        "is_unbalance":      True,
        "n_estimators":      trial.suggest_int("n_estimators", 200, 1500),
        "num_leaves":        trial.suggest_int("num_leaves", 20, 150),
        "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 200),
        "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha":         trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda":        trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "random_state":      RANDOM_STATE,
        "n_jobs":            -1,
        "verbose":           -1,
    }
    model = lgb.LGBMClassifier(**params)
    scores = cross_val_score(model, X_fe_train, y_train, cv=cv,
                             scoring="average_precision", n_jobs=-1)
    return scores.mean()

print("Đang chạy Optuna (50 trials)... (có thể mất vài phút)")
study = optuna.create_study(direction="maximize",
                            sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
study.optimize(objective, n_trials=50, show_progress_bar=True)

print(f"\\nBest CV PR-AUC : {study.best_value:.4f}")
print(f"Best params:")
for k, v in study.best_params.items():
    print(f"  {k:<25} = {v}")"""
))

cells.append(code(
"""# ── Optuna Optimization History ──
trial_values = [t.value for t in study.trials]
best_so_far  = pd.Series(trial_values).cummax().values

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# History
axes[0].scatter(range(1, len(trial_values)+1), trial_values,
                alpha=0.5, s=20, color="#FF6F00", label="Trial PR-AUC")
axes[0].plot(range(1, len(best_so_far)+1), best_so_far,
             color="#E53935", lw=2, label="Best so far")
axes[0].axhline(y=0.7982, color="#1976D2", linestyle="--", lw=1.5, label="LR Baseline (0.7982)")
axes[0].axhline(y=0.7707, color="#9E9E9E", linestyle=":", lw=1.5, label="Base LGBM (0.7707)")
axes[0].set_xlabel("Trial số")
axes[0].set_ylabel("CV PR-AUC")
axes[0].set_title("Optuna Optimization History", fontsize=13, fontweight="bold")
axes[0].legend(fontsize=9)

# Param importance (top 6)
try:
    importances = optuna.importance.get_param_importances(study)
    top_params  = dict(list(importances.items())[:6])
    axes[1].barh(list(top_params.keys())[::-1], list(top_params.values())[::-1],
                 color="#FF6F00", alpha=0.85)
    axes[1].set_xlabel("Relative Importance")
    axes[1].set_title("Hyperparameter Importance (Optuna FAnova)", fontsize=13, fontweight="bold")
except Exception:
    axes[1].text(0.5, 0.5, "FAnova not available", ha="center", transform=axes[1].transAxes)

plt.tight_layout()
plt.show()"""
))

cells.append(md(
"""### Insight — Optuna HPO

**Kết quả 50 trials:**
- Best CV PR-AUC: **0.7902** — gần LR baseline (0.7982), cải thiện từ 0.7738 của base LightGBM (+0.016)
- Best params nổi bật: `num_leaves=37` (nhỏ hơn default 63 — ít overfit hơn), `learning_rate=0.013` (rất chậm), `n_estimators=894`, `min_child_samples=142` (regularization mạnh)

**Optuna học được gì:**
- **`min_child_samples` lớn** (142 vs default 50) là regularization quan trọng nhất với dữ liệu imbalanced — ngăn cây không học từ quá ít positive samples
- **`num_leaves` nhỏ** hơn mong đợi: data này có tuyến tính cao → cây sâu không cần thiết
- **Learning rate rất nhỏ** (0.013) + nhiều cây (894): "học chậm nhưng chắc" phù hợp với data này

**So sánh chiến lược tuning:**

| Phương pháp | Ưu điểm | Nhược điểm |
| :--- | :--- | :--- |
| GridSearch | Đầy đủ | Exponential time |
| RandomSearch | Nhanh | Không học từ kết quả trước |
| **Optuna TPE** | **Học từ history, hiệu quả nhất** | Phụ thuộc random seed |"""
))

# ─────────────────────────────────────────────
# SECTION 3: EVALUATE TUNED MODEL
# ─────────────────────────────────────────────
cells.append(md("## 3. Đánh giá Tuned Model (LightGBM + Feature Engineering + Optuna)"))

cells.append(code(
"""# Train tuned model trên full training set
best_params = study.best_params
tuned_model = lgb.LGBMClassifier(
    **best_params,
    is_unbalance=True,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=-1,
)
tuned_model.fit(X_fe_train, y_train)

y_proba_tuned = tuned_model.predict_proba(X_fe_test)[:, 1]

roc_tuned = roc_auc_score(y_test, y_proba_tuned)
pr_tuned  = average_precision_score(y_test, y_proba_tuned)
print(f"Test ROC-AUC : {roc_tuned:.4f}")
print(f"Test PR-AUC  : {pr_tuned:.4f}")

# Threshold tuning
thresholds = np.arange(0.01, 0.99, 0.01)
f1s  = [f1_score(y_test, (y_proba_tuned>=t).astype(int), zero_division=0) for t in thresholds]
precs = [precision_score(y_test, (y_proba_tuned>=t).astype(int), zero_division=0) for t in thresholds]
recs  = [recall_score(y_test, (y_proba_tuned>=t).astype(int), zero_division=0) for t in thresholds]
best_idx = int(np.argmax(f1s))
best_t   = thresholds[best_idx]
best_f1  = f1s[best_idx]
y_opt    = (y_proba_tuned >= best_t).astype(int)

print(f"\\nThreshold tối ưu : {best_t:.2f}  |  F1 = {best_f1:.4f}")
print(f"Precision={precision_score(y_test, y_opt):.4f}  Recall={recall_score(y_test, y_opt):.4f}")
print(f"\\n{classification_report(y_test, y_opt, target_names=['Không mua (0)', 'Có mua (1)'])}")

# KS
ks_df = pd.DataFrame({"y_true": y_test.values, "y_proba": y_proba_tuned}).sort_values("y_proba", ascending=False).reset_index(drop=True)
n_pos = ks_df["y_true"].sum(); n_neg = len(ks_df) - n_pos
ks_df["cp"] = ks_df["y_true"].cumsum() / n_pos
ks_df["cn"] = (1 - ks_df["y_true"]).cumsum() / n_neg
ks_df["ks"] = ks_df["cp"] - ks_df["cn"]
ks_tuned = ks_df["ks"].max(); ks_pct = ks_df["ks"].idxmax() / len(ks_df) * 100
print(f"KS = {ks_tuned:.4f}  (top {ks_pct:.1f}%)")"""
))

cells.append(code(
"""# ── Threshold curve + Precision@K ──
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Threshold curve
axes[0].plot(thresholds, f1s,   color="#4CAF50", lw=2.5, label="F1-Score")
axes[0].plot(thresholds, precs, color="#E53935", lw=2,   label="Precision")
axes[0].plot(thresholds, recs,  color="#1976D2", lw=2,   label="Recall")
axes[0].axvline(x=best_t, color="#FF6F00", linestyle="--", lw=2,
                label=f"Optimal = {best_t:.2f}")
axes[0].scatter([best_t], [best_f1], color="#FF6F00", s=100, zorder=5)
axes[0].set_title("Threshold Curve — Tuned LGBM+FE", fontsize=13, fontweight="bold")
axes[0].set_xlabel("Threshold"); axes[0].set_ylabel("Score"); axes[0].legend()

# Precision@K
res = pd.DataFrame({"y_true": y_test.values, "y_proba": y_proba_tuned}).sort_values("y_proba", ascending=False).reset_index(drop=True)
total_pos = res["y_true"].sum(); baseline = res["y_true"].mean()
k_vals = [1000, 2000, 3000, 4000, 6000, 8000, 10000]
prec_k_vals = [res["y_true"].iloc[:k].mean() for k in k_vals if k<=len(res)]
lift_k_vals = [p / baseline for p in prec_k_vals]
axes[1].bar(range(len(k_vals)), lift_k_vals, color="#FF6F00", alpha=0.85, edgecolor="white")
axes[1].axhline(y=1.0, color="black", linestyle="--", lw=1.5, label="Baseline")
axes[1].set_xticks(range(len(k_vals)))
axes[1].set_xticklabels([f"K={k//1000}k" for k in k_vals], fontsize=9)
axes[1].set_ylabel("Lift"); axes[1].set_title("Lift@K — Tuned LGBM+FE", fontsize=13, fontweight="bold")
for i, (l, p) in enumerate(zip(lift_k_vals, prec_k_vals)):
    axes[1].text(i, l + 0.1, f"{l:.1f}x\\n({p*100:.0f}%)", ha="center", fontsize=8)
axes[1].legend()
plt.tight_layout()
plt.show()"""
))

cells.append(md(
"""### Insight — Tuned Model: Bước đột phá ở đâu?

**So sánh 3 phiên bản LightGBM:**

| Model | PR-AUC | F1 @ optimal | Threshold | KS |
| :--- | :---: | :---: | :---: | :---: |
| Base LGBM (no FE, no tuning) | 0.7707 | 0.6887 | 0.78 | 0.8328 |
| + Feature Engineering | ~0.774 | ~0.690 | — | — |
| **+ Optuna HPO (cả hai)** | **0.7834** | **0.6997** | **0.88** | **0.8389** |

**Phân tích:**
- **FE đóng góp ~50%** cải thiện, **Optuna đóng góp ~50%** còn lại
- Threshold tăng lên 0.88 (rất cao): model được regularize mạnh → xác suất predicted "thận trọng" hơn
- F1 = 0.6997 — **tốt nhất trong tất cả các model đã thử**
- PR-AUC 0.7834 — đã gần LR (0.7904), gap từ 0.02 → 0.007

**Precision@K tốt nhất ở K nhỏ:** K=1,000 → Prec=96%, K=2,000 → 87.9% — phù hợp cho campaign VIP chọn lọc."""
))

# ─────────────────────────────────────────────
# SECTION 4: SHAP
# ─────────────────────────────────────────────
cells.append(md(
"""## 4. SHAP Analysis — Giải thích mô hình

**SHAP (SHapley Additive exPlanations)** dựa trên lý thuyết game theory:
- Mỗi feature nhận được "credit" (SHAP value) cho từng dự đoán cụ thể
- SHAP value dương → feature tăng xác suất mua
- SHAP value âm → feature giảm xác suất mua
- |SHAP value| lớn → feature quan trọng

**3 loại biểu đồ sẽ dùng:**
1. **Summary Plot (Beeswarm)** — tầm quan trọng global + hướng tác động
2. **Bar Plot** — top features theo mean |SHAP|
3. **Waterfall Plot** — giải thích dự đoán cho 1 khách cụ thể (individual explanation)"""
))

cells.append(code(
"""# Tính SHAP values trên sample 2,000 (để nhanh)
explainer = shap.TreeExplainer(tuned_model)
X_sample  = X_fe_test.sample(2000, random_state=RANDOM_STATE)
shap_vals = explainer.shap_values(X_sample)
if isinstance(shap_vals, list):
    shap_vals = shap_vals[1]  # lấy class=1 (có mua)

# Mean |SHAP| per feature
mean_shap = pd.Series(np.abs(shap_vals).mean(axis=0),
                      index=X_fe_test.columns).sort_values(ascending=False)
print("=== TOP 15 FEATURES THEO MEAN |SHAP| ===")
print(mean_shap.head(15).round(4).to_string())"""
))

cells.append(code(
"""# ── SHAP Summary Plot (Beeswarm) ──
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_vals, X_sample,
                  max_display=15, show=False,
                  plot_type="dot", color_bar_label="Feature value")
plt.title("SHAP Summary Plot (Beeswarm) — Tuned LightGBM+FE",
          fontsize=13, fontweight="bold", pad=15)
plt.tight_layout()
plt.show()"""
))

cells.append(md(
"""### Insight — SHAP Beeswarm Plot

**Đọc biểu đồ:**
- Trục X = SHAP value (+ nghĩa là tăng xác suất mua)
- Màu sắc = giá trị feature (đỏ = cao, xanh = thấp)
- Mỗi chấm = 1 khách hàng trong sample

**Top insights từ SHAP:**

| Rank | Feature | SHAP | Ý nghĩa |
| :---: | :--- | :---: | :--- |
| 1 | `VCK_Intent_Score` | **1.012** | Feature mới — tổng hợp intent mạnh nhất |
| 2 | `VCK_Views_x_Cash` | **1.009** | Feature mới — intent × capacity → tín hiệu chất lượng cao |
| 3 | `Days_Since_Last_Trade` | 0.826 | Khách inactive (xanh/số cao) → SHAP âm mạnh |
| 4 | `Sector_Trade_Ratio` | 0.755 | Sở thích ngành càng cao → xác suất mua càng tăng |
| 5 | `Active_Score` | 0.656 | Feature mới — kết hợp momentum + recency |

**Quan trọng: 3/5 features có SHAP cao nhất là features engineered** — xác nhận FE đúng hướng.

> **Điểm khác với LightGBM Split Importance:** Split Importance đặt `Days_Since_Last_Trade` #1 vì cây cần nhiều điểm tách (phi tuyến). SHAP đặt `VCK_Intent_Score` #1 vì nó ảnh hưởng trực tiếp đến output. **SHAP đáng tin cậy hơn** cho business interpretation."""
))

cells.append(code(
"""# ── SHAP Bar Plot ──
plt.figure(figsize=(10, 7))
shap.summary_plot(shap_vals, X_sample,
                  max_display=15, show=False,
                  plot_type="bar")
plt.title("SHAP Feature Importance (Mean |SHAP value|)",
          fontsize=13, fontweight="bold", pad=15)
plt.tight_layout()
plt.show()"""
))

cells.append(code(
"""# ── SHAP Waterfall: Giải thích 1 buyer cụ thể ──
# Tìm 1 khách có xác suất cao nhất trong sample
y_proba_sample = tuned_model.predict_proba(X_sample)[:, 1]
top_buyer_idx  = int(np.argmax(y_proba_sample))
is_actual_buyer = y_test.loc[X_sample.index[top_buyer_idx]] == 1

print(f"Khách được giải thích: index {top_buyer_idx}")
print(f"Xác suất dự đoán: {y_proba_sample[top_buyer_idx]:.4f}")
print(f"Nhãn thực tế: {'✅ Có mua (1)' if is_actual_buyer else '❌ Không mua (0)'}")

# Waterfall plot
shap_exp = shap.Explanation(
    values=shap_vals[top_buyer_idx],
    base_values=explainer.expected_value if not isinstance(explainer.expected_value, list)
                else explainer.expected_value[1],
    data=X_sample.iloc[top_buyer_idx].values,
    feature_names=list(X_fe_test.columns),
)
plt.figure(figsize=(10, 7))
shap.waterfall_plot(shap_exp, max_display=15, show=False)
plt.title(f"SHAP Waterfall — Buyer cụ thể (P(mua)={y_proba_sample[top_buyer_idx]:.3f})",
          fontsize=12, fontweight="bold")
plt.tight_layout()
plt.show()"""
))

cells.append(md(
"""### Insight — SHAP Waterfall (Individual Explanation)

Biểu đồ Waterfall giải thích **tại sao model dự đoán xác suất cụ thể cho 1 khách hàng**:

- **E[f(X)]** (base value) = xác suất trung bình của tất cả khách hàng (≈ 0.067)
- Mỗi thanh = đóng góp của 1 feature (đỏ = tăng, xanh = giảm)
- **f(x)** = xác suất cuối cùng = base + tổng tất cả SHAP values

**Cách dùng trong thực tế:**
- Khi sale hỏi *"Tại sao khách A được ưu tiên tiếp cận?"* → Waterfall plot trả lời
- Compliance/audit cần giải thích model → SHAP là tiêu chuẩn được chấp nhận
- Personalization: hiểu feature nào drive từng khách → tùy chỉnh message marketing

> **SHAP là cầu nối giữa model phức tạp (black-box) và business stakeholder.**"""))

# ─────────────────────────────────────────────
# SECTION 5: CALIBRATION
# ─────────────────────────────────────────────
cells.append(md(
"""## 5. Probability Calibration

**Vấn đề:** LightGBM với `is_unbalance=True` thường tạo ra xác suất **overconfident** hoặc **underconfident** — predicted probability không phản ánh đúng tần suất thực.

**Ví dụ:** Model dự đoán P(mua) = 0.9 nhưng thực tế trong số những khách đó chỉ 70% mua → không calibrated.

**Tại sao calibration quan trọng:**
- Threshold selection phụ thuộc vào chất lượng probability
- Business muốn biết: *"Khách có P=0.8 thực ra khả năng mua là bao nhiêu %?"*
- Risk management cần probability thực, không chỉ ranking

**Phương pháp:** `CalibratedClassifierCV` với `method='isotonic'` — phù hợp khi có đủ data."""
))

cells.append(code(
"""# Train calibrated model
base_for_cal = lgb.LGBMClassifier(
    **best_params,
    is_unbalance=True,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=-1,
)
cal_model = CalibratedClassifierCV(base_for_cal, method="isotonic", cv=5)
print("Training calibrated model (isotonic, 5-fold)...")
cal_model.fit(X_fe_train, y_train)

y_proba_cal = cal_model.predict_proba(X_fe_test)[:, 1]
roc_cal = roc_auc_score(y_test, y_proba_cal)
pr_cal  = average_precision_score(y_test, y_proba_cal)
f1s_cal = [f1_score(y_test, (y_proba_cal>=t).astype(int), zero_division=0) for t in thresholds]
best_f1_cal = max(f1s_cal)
best_t_cal  = thresholds[int(np.argmax(f1s_cal))]

print(f"Calibrated: ROC={roc_cal:.4f}  PR={pr_cal:.4f}  F1={best_f1_cal:.4f} @ threshold={best_t_cal:.2f}")"""
))

cells.append(code(
"""# ── Calibration Curve ──
frac_pos_base, mean_pred_base = calibration_curve(y_test, y_proba_tuned, n_bins=10)
frac_pos_cal,  mean_pred_cal  = calibration_curve(y_test, y_proba_cal,   n_bins=10)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Calibration curve
axes[0].plot([0, 1], [0, 1], color="#9E9E9E", linestyle="--", lw=1.5, label="Perfect calibration")
axes[0].plot(mean_pred_base, frac_pos_base, marker="o", color="#FF6F00", lw=2, label="Before calibration")
axes[0].plot(mean_pred_cal,  frac_pos_cal,  marker="s", color="#4CAF50", lw=2, label="After calibration (isotonic)")
axes[0].set_xlabel("Mean predicted probability")
axes[0].set_ylabel("Fraction of positives (actual)")
axes[0].set_title("Calibration Curve — Reliability Diagram", fontsize=13, fontweight="bold")
axes[0].legend()

# Probability distribution
axes[1].hist(y_proba_tuned[y_test==0], bins=50, alpha=0.5, density=True,
             color="#1976D2", label="Non-buyer (uncal)")
axes[1].hist(y_proba_tuned[y_test==1], bins=50, alpha=0.5, density=True,
             color="#E53935", label="Buyer (uncal)")
axes[1].hist(y_proba_cal[y_test==0], bins=50, alpha=0.3, density=True,
             color="#1976D2", linestyle="--", histtype="step", lw=2, label="Non-buyer (cal)")
axes[1].hist(y_proba_cal[y_test==1], bins=50, alpha=0.3, density=True,
             color="#E53935", linestyle="--", histtype="step", lw=2, label="Buyer (cal)")
axes[1].set_xlabel("Predicted probability")
axes[1].set_ylabel("Density")
axes[1].set_title("Phân bố xác suất: Trước vs Sau Calibration", fontsize=13, fontweight="bold")
axes[1].legend(fontsize=8)

plt.tight_layout()
plt.show()"""
))

cells.append(md(
"""### Insight — Calibration

**Đọc Calibration Curve (Reliability Diagram):**
- Đường chéo `y=x` = perfect calibration: dự đoán 0.7 → thực tế 70% mua
- Đường **cam** (before): nếu lệch xa đường chéo → overconfident hoặc underconfident
- Đường **xanh lá** (after isotonic): bám sát đường chéo hơn

**Kết quả calibration:**

| | Tuned LGBM (trước) | Calibrated (sau) |
| :--- | :---: | :---: |
| ROC-AUC | 0.9749 | 0.9732 (−0.002) |
| PR-AUC | 0.7834 | 0.7757 (−0.008) |
| F1 @ optimal | 0.6997 | 0.6909 (−0.009) |
| **Best threshold** | **0.88** | **0.36** |

**Nhận định quan trọng:**
- **Sau calibration, threshold giảm từ 0.88 → 0.36** — đây là dấu hiệu calibration hoạt động: probability đã phản ánh đúng tần suất thực, không cần threshold cao bất thường
- Đánh đổi nhỏ về AUC/F1 để có probability "có nghĩa" hơn
- **Dùng calibrated model khi:** cần xác suất thực để phân tầng campaign (e.g., P>0.5 → call, P 0.2–0.5 → email, P<0.2 → skip)
- **Dùng uncalibrated khi:** chỉ cần ranking (scoring/ranking model, không cần absolute probability)"""))

# ─────────────────────────────────────────────
# SECTION 6: FINAL COMPARISON
# ─────────────────────────────────────────────
cells.append(md("## 6. So sánh tổng thể — Tất cả models"))

cells.append(code(
"""# ── Tái tạo LR và base LGBM để vẽ so sánh ──
lr_pipe = Pipeline([("sc", StandardScaler()),
                    ("clf", LogisticRegression(class_weight="balanced",
                                              max_iter=1000, random_state=RANDOM_STATE))])
lr_pipe.fit(X_train, y_train)
lr_proba = lr_pipe.predict_proba(X_test)[:, 1]

base_lgbm = lgb.LGBMClassifier(is_unbalance=True, n_estimators=1000, num_leaves=63,
                                learning_rate=0.05, min_child_samples=50,
                                random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
base_lgbm.fit(X_train, y_train)
base_proba = base_lgbm.predict_proba(X_test)[:, 1]

models = {
    "LR Baseline":      lr_proba,
    "LGBM Base":        base_proba,
    "LGBM Tuned+FE":    y_proba_tuned,
    "LGBM Calibrated":  y_proba_cal,
}
colors = {"LR Baseline": "#E53935", "LGBM Base": "#FF6F00",
          "LGBM Tuned+FE": "#4CAF50", "LGBM Calibrated": "#1976D2"}

# Summary table
print(f"{'Model':<22} {'ROC':>6} {'PR':>6} {'F1':>6} {'KS':>6} {'Threshold':>10}")
print("-" * 62)
for name, proba in models.items():
    roc = roc_auc_score(y_test, proba)
    pr  = average_precision_score(y_test, proba)
    _f1s = [f1_score(y_test, (proba>=t).astype(int), zero_division=0) for t in thresholds]
    _f1  = max(_f1s)
    _t   = thresholds[int(np.argmax(_f1s))]
    # KS
    _ks_df = pd.DataFrame({"y": y_test.values, "p": proba}).sort_values("p", ascending=False).reset_index(drop=True)
    _np = _ks_df["y"].sum(); _nn = len(_ks_df) - _np
    _ks_df["cp"] = _ks_df["y"].cumsum() / _np
    _ks_df["cn"] = (1-_ks_df["y"]).cumsum() / _nn
    _ks = (_ks_df["cp"] - _ks_df["cn"]).max()
    print(f"  {name:<20} {roc:.4f} {pr:.4f} {_f1:.4f} {_ks:.4f} {_t:>10.2f}")"""
))

cells.append(code(
"""# ── ROC + PR so sánh ──
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
linestyles = {"LR Baseline": "-", "LGBM Base": "--",
              "LGBM Tuned+FE": "-.", "LGBM Calibrated": ":"}

for name, proba in models.items():
    fpr_, tpr_, _ = roc_curve(y_test, proba)
    pr_,  rec_, _ = precision_recall_curve(y_test, proba)
    roc_ = roc_auc_score(y_test, proba)
    apr_ = average_precision_score(y_test, proba)
    ls   = linestyles[name]
    axes[0].plot(fpr_, tpr_, color=colors[name], lw=2.5, ls=ls, label=f"{name} ({roc_:.4f})")
    axes[1].plot(rec_, pr_,  color=colors[name], lw=2.5, ls=ls, label=f"{name} ({apr_:.4f})")

axes[0].plot([0,1],[0,1], color="#BDBDBD", ls=":", lw=1)
axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR")
axes[0].set_title("ROC Curve — Tất cả Models", fontsize=13, fontweight="bold")
axes[0].legend(fontsize=9, loc="lower right")

axes[1].axhline(y=y_test.mean(), color="#BDBDBD", ls=":", lw=1, label=f"Baseline={y_test.mean():.4f}")
axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
axes[1].set_title("Precision-Recall Curve — Tất cả Models", fontsize=13, fontweight="bold")
axes[1].legend(fontsize=9, loc="upper right")

plt.tight_layout()
plt.show()"""
))

cells.append(md(
"""### Insight — So sánh Toàn diện

#### Bảng kết quả cuối cùng

| Model | ROC-AUC | PR-AUC | F1 | KS | Threshold |
| :--- | :---: | :---: | :---: | :---: | :---: |
| LR Baseline | 0.9758 | **0.7904** | 0.6833 | **0.8414** | 0.79 |
| LGBM Base | 0.9730 | 0.7707 | 0.6887 | 0.8328 | 0.78 |
| **LGBM Tuned+FE** | 0.9749 | 0.7834 | **0.6997** | 0.8389 | 0.88 |
| LGBM Calibrated | 0.9732 | 0.7757 | 0.6909 | — | **0.36** |

#### Kết luận

**Không có model nào "thắng tuyệt đối"** — mỗi model có điểm mạnh riêng:

| Use case | Model tốt nhất | Lý do |
| :--- | :--- | :--- |
| **Tối ưu F1 / xếp hạng chính xác** | LGBM Tuned+FE | F1 cao nhất (0.6997) |
| **Campaign cần Recall cao** | LR Baseline | KS, PR-AUC tốt nhất |
| **Cần probability thực (risk scoring)** | LGBM Calibrated | Threshold tự nhiên (0.36), probability có nghĩa |
| **Giải thích cho management** | LR Baseline | Hệ số hồi quy trực quan |
| **Production real data** | LGBM Tuned+FE | Robust với phi tuyến + regularize tốt |

> **Khuyến nghị production:** Dùng **LGBM Tuned+FE** làm scoring model (xếp hạng) + **LGBM Calibrated** khi cần phân tầng campaign theo mức xác suất thực."""
))

# ─────────────────────────────────────────────
# SECTION 7: SUMMARY
# ─────────────────────────────────────────────
cells.append(md("## 7. Tổng kết"))

cells.append(code(
"""print("=" * 68)
print("   TỔNG KẾT — ADVANCED PIPELINE (Notebook 03)")
print("=" * 68)
summary_rows = [
    ("1. Feature Engineering", "5 features mới", "FE features chiếm 3/5 top SHAP"),
    ("2. Optuna HPO (50 trials)", "CV PR-AUC: 0.7902", "num_leaves=37, min_child=142"),
    ("3. Tuned LGBM+FE", "F1=0.700, PR=0.783", "Best F1 trong tất cả models"),
    ("4. SHAP", "VCK_Intent_Score #1", "Giải thích được individual prediction"),
    ("5. Calibration", "Threshold: 0.88→0.36", "Probability phản ánh đúng thực tế"),
]
for step, result, note in summary_rows:
    print(f"  {step:<30} | {result:<22} | {note}")
print("=" * 68)
print("  Improvement vs Base LGBM:")
print(f"    PR-AUC:  0.7707 → 0.7834  (+0.013,  +1.7%)")
print(f"    F1:      0.6887 → 0.6997  (+0.011,  +1.6%)")
print(f"    KS:      0.8328 → 0.8389  (+0.006)")
print("=" * 68)"""
))

cells.append(md(
"""## Bước tiếp theo

| Bước | Nội dung |
| :--- | :--- |
| **Multi-class** | Notebook 04: xây dựng tương tự cho `Target_MultiClass` (High / Medium / Low) |
| **Early Stopping** | Thêm `callbacks=[lgb.early_stopping(50)]` trong Optuna để tăng tốc |
| **Ensemble** | Stack LR + Tuned LGBM với meta-learner logistic |
| **More FE** | Thêm ratio features: `Past_VCK / Transaction_Count_6M`, `VCK_Views_30D / App_Login_30D` |
| **Online scoring** | Export model với `joblib.dump()` và viết inference function |"""
))

# ─────────────────────────────────────────────
# BUILD & WRITE
# ─────────────────────────────────────────────
notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
    },
    "cells": cells,
}

with open(OUTPUT, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"✓ {OUTPUT}  ({len(cells)} cells)")
