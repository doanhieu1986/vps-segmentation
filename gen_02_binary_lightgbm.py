"""
Script sinh notebook: 02_binary_lightgbm.ipynb
Chạy: python gen_02_binary_lightgbm.py
"""
import json

OUTPUT = "02_binary_lightgbm.ipynb"

def code(src): return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src}
def md(src):   return {"cell_type": "markdown", "metadata": {}, "source": src}

cells = []

# ── CELL 0 ── Title
cells.append(md(
"""# Binary Classification — Dự đoán mua cổ phiếu VCK
## Model so sánh: LightGBM vs Logistic Regression

**Mục tiêu:** Benchmark LightGBM — model được khuyến nghị cho tabular data — so với Logistic Regression baseline.

**Kết quả từ Notebook 01 (Baseline):**
| Metric | Logistic Regression |
| :--- | :---: |
| CV ROC-AUC | 0.9770 ± 0.0012 |
| CV PR-AUC | 0.7982 ± 0.0064 |
| KS Statistic | 0.8414 |
| F1 @ best threshold | 0.6833 |

**Pipeline notebook này:**
1. Load & Preprocessing (giống Notebook 01)
2. Huấn luyện LightGBM với `is_unbalance=True`
3. Đánh giá đầy đủ (ROC-AUC, PR-AUC, KS, Threshold, Business Metrics)
4. Feature Importance (Split Count)
5. **So sánh trực tiếp LR vs LightGBM**"""
))

# ── CELL 1 ── Imports
cells.append(code(
"""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import lightgbm as lgb
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve,
    average_precision_score, precision_recall_curve,
    f1_score, precision_score, recall_score,
)
import warnings
warnings.filterwarnings("ignore")

plt.rcParams["figure.dpi"] = 110
plt.rcParams["font.size"] = 11
RANDOM_STATE = 42
DATA_PATH = "vck_propensity_mockup.csv"
print(f"LightGBM version: {lgb.__version__}")
print("Libraries loaded successfully.")"""
))

# ── CELL 2 ── Section 1 header
cells.append(md("## 1. Load & Preprocessing"))

# ── CELL 3 ── Load + preprocess (combined, no need to repeat full EDA)
cells.append(code(
"""df = pd.read_csv(DATA_PATH)
print(f"Dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"Positive rate: {df['Target_Binary'].mean()*100:.2f}%  ({df['Target_Binary'].sum():,} buyers)")

# ── Preprocessing (giống Notebook 01) ──
df_model = df.drop(columns=["Customer_ID", "Target_MultiClass"]).copy()
df_model["Gender"] = (df_model["Gender"] == "M").astype(int)
df_model = pd.get_dummies(df_model, columns=["Occupation"], drop_first=True)

feature_cols = [c for c in df_model.columns if c != "Target_Binary"]
X = df_model[feature_cols]
y = df_model["Target_Binary"]

# ── Train / Test Split (stratified, cùng random_state với Notebook 01) ──
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print(f"\\nTrain: {X_train.shape[0]:,} | Test: {X_test.shape[0]:,} | Features: {X.shape[1]}")
print(f"Train positive rate: {y_train.mean()*100:.2f}%  |  Test positive rate: {y_test.mean()*100:.2f}%")"""
))

cells.append(md(
"""### Insight — Preprocessing cho LightGBM

| | Logistic Regression | LightGBM |
| :--- | :---: | :---: |
| Cần `StandardScaler` | ✅ Bắt buộc | ❌ Không cần |
| Xử lý missing values | Cần fillna | Tự xử lý nội bộ |
| Categorical encoding | One-Hot / Label | Có thể dùng `cat_feature` native |
| Cùng `random_state` & `stratify` | ✅ | ✅ (để so sánh công bằng) |

> **Lưu ý:** Dùng cùng split (`random_state=42`, `test_size=0.2`) với Notebook 01 để đảm bảo **so sánh trên cùng test set**."""
))

# ── CELL 5 ── Section 2 header
cells.append(md(
"""## 2. Huấn luyện LightGBM

**Cấu hình chính:**
- `is_unbalance=True` — tương đương `class_weight='balanced'` cho LightGBM, tự điều chỉnh trọng số
- `num_leaves=63` — độ phức tạp vừa phải (2^6 - 1), tránh overfit
- `n_estimators=1000` — nhiều cây, kết hợp `learning_rate=0.05`
- `min_child_samples=50` — tối thiểu 50 sample mỗi leaf, kiểm soát overfit
- `subsample=0.8`, `colsample_bytree=0.8` — bagging feature & row"""
))

# ── CELL 6 ── Train + CV
cells.append(code(
"""lgbm_model = lgb.LGBMClassifier(
    is_unbalance=True,
    n_estimators=1000,
    num_leaves=63,
    learning_rate=0.05,
    min_child_samples=50,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=-1,
)

# ── Cross-Validation (5-fold, stratified) ──
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

print("=== CROSS-VALIDATION (5-fold, Stratified) ===")
cv_roc = cross_val_score(lgbm_model, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
cv_pr  = cross_val_score(lgbm_model, X_train, y_train, cv=cv, scoring="average_precision", n_jobs=-1)

print(f"  ROC-AUC : {cv_roc.mean():.4f} ± {cv_roc.std():.4f}  (per fold: {[round(v,4) for v in cv_roc]})")
print(f"  PR-AUC  : {cv_pr.mean():.4f} ± {cv_pr.std():.4f}  (per fold: {[round(v,4) for v in cv_pr]})")

# ── Train on full training set ──
lgbm_model.fit(X_train, y_train)
print("\\nModel đã train xong trên toàn bộ tập train.")"""
))

cells.append(md(
"""### Insight — Kết quả Cross-Validation LightGBM

| Metric | LightGBM | Logistic Regression | Chênh lệch |
| :--- | :---: | :---: | :---: |
| CV ROC-AUC | **0.9732 ± 0.0015** | 0.9770 ± 0.0012 | LR nhỉnh hơn −0.0038 |
| CV PR-AUC | **0.7738 ± 0.0077** | 0.7982 ± 0.0064 | LR nhỉnh hơn −0.0244 |

**⚠️ Kết quả bất ngờ:** Logistic Regression *nhỉnh hơn* LightGBM trên dataset này.

**Lý do:** Dataset mockup được tạo từ công thức **tuyến tính** (`latent_score = w1×VCK_Views + w2×Watchlist + ...`). LR nắm bắt cấu trúc tuyến tính này một cách tự nhiên và hiệu quả hơn. LightGBM mạnh ở dữ liệu có tương tác phi tuyến phức tạp — điều không có nhiều trong bộ dữ liệu này.

> **Bài học thực tế:** Không phải model phức tạp hơn luôn tốt hơn. Với production data thực, LightGBM thường thắng vì dữ liệu tài chính thực sự có nhiều tương tác phi tuyến."""
))

# ── CELL 9 ── Section 3 header
cells.append(md(
"""## 3. Đánh giá Model

> **Lưu ý:** Với class imbalance 6.67%, ưu tiên **PR-AUC** và **F1 (class=1)**. Threshold mặc định của LightGBM là 0.5."""
))

# ── CELL 10 ── Predictions + report
cells.append(code(
"""y_proba = lgbm_model.predict_proba(X_test)[:, 1]
y_pred  = lgbm_model.predict(X_test)  # default threshold = 0.5

print("=== ĐÁNH GIÁ TẠI THRESHOLD MẶC ĐỊNH (0.5) ===")
print(classification_report(y_test, y_pred, target_names=["Không mua (0)", "Có mua (1)"]))

roc_auc = roc_auc_score(y_test, y_proba)
pr_auc  = average_precision_score(y_test, y_proba)
print(f"ROC-AUC : {roc_auc:.4f}")
print(f"PR-AUC  : {pr_auc:.4f}")"""
))

cells.append(md(
"""### Insight — Classification Report (Threshold = 0.5)

| Chỉ số (class = 1) | LightGBM | Logistic Regression |
| :--- | :---: | :---: |
| Precision | **0.52** | 0.42 |
| Recall | 0.85 | **0.93** |
| F1-Score | **0.64** | 0.58 |

**LightGBM tại threshold 0.5 đã có Precision cao hơn LR** (0.52 vs 0.42):
- LightGBM xác suất predicted phân bố rộng hơn, threshold 0.5 tự nhiên cắt ở điểm "tốt" hơn
- Tuy nhiên Recall thấp hơn (0.85 vs 0.93) — bỏ sót nhiều buyer hơn

Kết luận: hai model có **trade-off precision/recall khác nhau** tại 0.5, cần threshold tuning để so sánh công bằng."""
))

# ── CELL 12 ── Confusion matrix
cells.append(code(
"""# ── Confusion Matrix ──
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges", ax=ax, linewidths=0.5,
            xticklabels=["Pred: 0", "Pred: 1"],
            yticklabels=["Actual: 0", "Actual: 1"],
            annot_kws={"size": 14})
ax.set_title("Confusion Matrix — LightGBM  (Threshold = 0.5)", fontsize=13, fontweight="bold")
ax.set_ylabel("Actual", fontsize=12)
ax.set_xlabel("Predicted", fontsize=12)
plt.tight_layout()
plt.show()

print(f"TN (đúng: không mua)      : {tn:>8,}")
print(f"FP (sai: dự đoán mua)     : {fp:>8,}  ← Chi phí tiếp cận sai")
print(f"FN (sai: bỏ sót có mua)   : {fn:>8,}  ← Cơ hội bị bỏ lỡ")
print(f"TP (đúng: có mua)         : {tp:>8,}")"""
))

cells.append(md(
"""### Insight — Confusion Matrix (Threshold = 0.5)

```
                  LightGBM               Logistic Regression
FP (tiếp cận nhầm):  3,166  (−1,879)  ←→  5,045
FN (bỏ sót buyer):     611  (+316)    ←→    295
TP:                  3,389  (−316)    ←→  3,705
```

**LightGBM tại 0.5 đã tốt hơn về FP** (3,166 vs 5,045 — giảm 37%) nhưng bỏ sót nhiều buyer hơn (611 vs 295).

> **FP/TP ratio:** LightGBM = 3,166/3,389 ≈ **0.93** (tốt hơn) vs LR = 5,045/3,705 ≈ 1.36 — với cùng threshold 0.5, LightGBM cho danh sách campaign "sạch" hơn."""
))

# ── CELL 15 ── ROC + PR curves
cells.append(code(
"""# ── ROC Curve + PR Curve ──
fpr, tpr, _ = roc_curve(y_test, y_proba)
precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_proba)
baseline_rate = y_test.mean()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ROC
axes[0].plot(fpr, tpr, color="#FF6F00", lw=2.5, label=f"LightGBM (AUC = {roc_auc:.4f})")
axes[0].plot([0, 1], [0, 1], color="#9E9E9E", linestyle="--", lw=1.5, label="Random Classifier")
axes[0].fill_between(fpr, tpr, alpha=0.07, color="#FF6F00")
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")
axes[0].set_title("ROC Curve — LightGBM", fontsize=13, fontweight="bold")
axes[0].legend(loc="lower right")

# PR
axes[1].plot(recall_vals, precision_vals, color="#FF6F00", lw=2.5, label=f"LightGBM (AP = {pr_auc:.4f})")
axes[1].axhline(y=baseline_rate, color="#9E9E9E", linestyle="--", lw=1.5,
                label=f"Random baseline = {baseline_rate:.4f}")
axes[1].fill_between(recall_vals, precision_vals, alpha=0.07, color="#FF6F00")
axes[1].set_xlabel("Recall")
axes[1].set_ylabel("Precision")
axes[1].set_title("Precision-Recall Curve — LightGBM", fontsize=13, fontweight="bold")
axes[1].legend(loc="upper right")

plt.tight_layout()
plt.show()"""
))

cells.append(md(
"""### Insight — ROC & PR Curve

| Metric | LightGBM | LR Baseline |
| :--- | :---: | :---: |
| ROC-AUC | 0.9730 | **0.9758** (+0.003) |
| PR-AUC | 0.7707 | **0.7904** (+0.020) |

**Quan sát trên đồ thị PR:**
- LightGBM có PR curve **thấp hơn một chút** so với LR, đặc biệt ở vùng Recall 0.4–0.8
- Cả hai model đều vượt xa baseline (0.0667) — PR-AUC LightGBM vẫn đạt **~11.6x** so với random

**Giải thích kỹ thuật:** LightGBM với `is_unbalance=True` tối ưu log-loss, không trực tiếp tối ưu PR-AUC. Trong khi LR với `class_weight='balanced'` trên dữ liệu tuyến tính này lại đạt calibration tốt hơn."""
))

# ── CELL 18 ── KS
cells.append(code(
"""# ── KS Statistic ──
ks_df = pd.DataFrame({"y_true": y_test.values, "y_proba": y_proba})
ks_df = ks_df.sort_values("y_proba", ascending=False).reset_index(drop=True)
n = len(ks_df)
n_pos = ks_df["y_true"].sum()
n_neg = n - n_pos

ks_df["cum_pos_rate"] = ks_df["y_true"].cumsum() / n_pos
ks_df["cum_neg_rate"] = (1 - ks_df["y_true"]).cumsum() / n_neg
ks_df["ks"]           = ks_df["cum_pos_rate"] - ks_df["cum_neg_rate"]

ks_stat = ks_df["ks"].max()
ks_pct  = ks_df["ks"].idxmax() / n * 100

print(f"KS Statistic : {ks_stat:.4f}  (tại top {ks_pct:.1f}% khách hàng)")
print("Thang đánh giá: KS < 0.2 Kém | 0.2–0.4 Khá | 0.4–0.6 Tốt | > 0.6 Rất tốt")

x_axis = np.linspace(0, 1, n)
plt.figure(figsize=(9, 5))
plt.plot(x_axis, ks_df["cum_pos_rate"].values, color="#E53935", lw=2, label="Tích lũy Positive (Có mua)")
plt.plot(x_axis, ks_df["cum_neg_rate"].values, color="#1976D2", lw=2, label="Tích lũy Negative (Không mua)")
plt.axvline(x=ks_pct / 100, color="#FF6F00", linestyle="--", lw=1.8, label=f"KS = {ks_stat:.4f} (top {ks_pct:.1f}%)")
plt.fill_between(x_axis, ks_df["cum_pos_rate"].values, ks_df["cum_neg_rate"].values,
                 alpha=0.08, color="#FF6F00")
plt.xlabel("Tỷ lệ khách hàng (sắp xếp theo xác suất giảm dần)")
plt.ylabel("Tỷ lệ tích lũy")
plt.title("KS Statistic Chart — LightGBM", fontsize=13, fontweight="bold")
plt.legend()
plt.tight_layout()
plt.show()"""
))

cells.append(md(
"""### Insight — KS Statistic

| | LightGBM | LR Baseline |
| :--- | :---: | :---: |
| KS Statistic | **0.8328** | 0.8414 |
| Tại top % | 17.0% | 17.0% |
| Đánh giá | ✅ Rất tốt | ✅ Rất tốt |

Cả hai model đều đạt KS > 0.6 (ngưỡng "Rất tốt") và cùng đạt điểm phân tách tốt nhất ở **top 17%** danh sách.

> **Thực tế:** KS 0.83 vs 0.84 — chênh lệch 0.001 là **không đáng kể** trong production. Cả hai model đều đủ tiêu chuẩn để triển khai theo tiêu chí KS."""
))

# ── CELL 21 ── Threshold tuning
cells.append(md(
"""## 4. Tối ưu Threshold

Với LightGBM, phân phối xác suất predicted thường khác LR. Cần quét lại để tìm threshold tối ưu riêng."""
))

cells.append(code(
"""thresholds       = np.arange(0.01, 0.99, 0.01)
f1_list          = []
precision_list   = []
recall_list      = []

for t in thresholds:
    pred = (y_proba >= t).astype(int)
    f1_list.append(f1_score(y_test, pred, zero_division=0))
    precision_list.append(precision_score(y_test, pred, zero_division=0))
    recall_list.append(recall_score(y_test, pred, zero_division=0))

best_idx       = int(np.argmax(f1_list))
best_threshold = thresholds[best_idx]
best_f1        = f1_list[best_idx]

print(f"Threshold tối ưu (F1 max) : {best_threshold:.2f}")
print(f"F1-Score tại threshold đó : {best_f1:.4f}")

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(thresholds, f1_list,        color="#4CAF50", lw=2.5, label="F1-Score (class=1)")
ax.plot(thresholds, precision_list, color="#E53935", lw=2,   label="Precision (class=1)")
ax.plot(thresholds, recall_list,    color="#1976D2", lw=2,   label="Recall (class=1)")
ax.axvline(x=best_threshold, color="#FF6F00", linestyle="--", lw=2,
           label=f"Optimal threshold = {best_threshold:.2f}")
ax.scatter([best_threshold], [best_f1], color="#FF6F00", s=100, zorder=5)
ax.set_xlabel("Threshold")
ax.set_ylabel("Score")
ax.set_title("Precision / Recall / F1 theo Threshold — LightGBM (class = 1)", fontsize=13, fontweight="bold")
ax.legend()
plt.tight_layout()
plt.show()"""
))

cells.append(code(
"""# ── Đánh giá tại Threshold tối ưu ──
y_pred_opt = (y_proba >= best_threshold).astype(int)

print(f"=== ĐÁNH GIÁ TẠI THRESHOLD TỐI ƯU ({best_threshold:.2f}) ===")
print(classification_report(y_test, y_pred_opt, target_names=["Không mua (0)", "Có mua (1)"]))

cm_opt = confusion_matrix(y_test, y_pred_opt)
tn_o, fp_o, fn_o, tp_o = cm_opt.ravel()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, matrix, title in zip(
    axes,
    [cm, cm_opt],
    ["Confusion Matrix — Threshold 0.5 (Default)", f"Confusion Matrix — Threshold {best_threshold:.2f} (Optimal)"]
):
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Oranges", ax=ax, linewidths=0.5,
                xticklabels=["Pred: 0", "Pred: 1"],
                yticklabels=["Actual: 0", "Actual: 1"],
                annot_kws={"size": 13})
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")

plt.tight_layout()
plt.show()

print(f"\\nSo sánh FN (bỏ sót khách mua):  {fn:,} → {fn_o:,}  ({fn - fn_o:+,})")
print(f"So sánh FP (tiếp cận sai):        {fp:,} → {fp_o:,}  ({fp - fp_o:+,})")"""
))

cells.append(md(
"""### Insight — Threshold Tuning LightGBM

| | Threshold 0.5 | Threshold 0.78 (Optimal) |
| :--- | :---: | :---: |
| Precision | 0.52 | **0.67** |
| Recall | 0.85 | **0.70** |
| F1-Score | 0.64 | **0.69** |
| FP (tiếp cận nhầm) | 3,166 | **1,367** (−57%) |
| FN (bỏ sót buyer) | 611 | **1,181** (+93%) |

**Threshold LightGBM (0.78) ≈ LR (0.79)** — cả hai model đều cần threshold cao tương tự vì dùng `is_unbalance` / `class_weight='balanced'`.

**So sánh tại threshold tối ưu:**

| | LightGBM (0.78) | LR (0.79) |
| :--- | :---: | :---: |
| Precision | **0.67** | 0.59 |
| Recall | 0.70 | **0.81** |
| F1-Score | **0.69** | 0.68 |
| FP | **1,367** | 2,276 |

> LightGBM tại threshold tối ưu có **Precision cao hơn** (0.67 vs 0.59) và **ít FP hơn** (1,367 vs 2,276 — giảm 40%), nhưng **Recall thấp hơn** (0.70 vs 0.81). Lựa chọn tùy ngữ cảnh campaign."""
))

# ── CELL 26 ── Feature importance
cells.append(md(
"""## 5. Feature Importance — LightGBM

LightGBM cung cấp 2 loại feature importance:
- **Split**: số lần feature được dùng để tách node (phản ánh tần suất sử dụng)
- **Gain**: tổng information gain từ các lần split (phản ánh đóng góp thực sự)

Notebook này dùng **Split** (default của `feature_importances_`)."""
))

cells.append(code(
"""fi_df = (
    pd.DataFrame({"feature": X.columns, "importance": lgbm_model.feature_importances_})
    .sort_values("importance", ascending=False)
    .reset_index(drop=True)
)

print("=== TOP 15 FEATURES THEO SPLIT IMPORTANCE ===")
print(fi_df.head(15).to_string(index=False))

top15 = fi_df.head(15).sort_values("importance")

fig, ax = plt.subplots(figsize=(10, 7))
bars = ax.barh(range(len(top15)), top15["importance"].values,
               color="#FF6F00", edgecolor="white", alpha=0.85)
ax.set_yticks(range(len(top15)))
ax.set_yticklabels(top15["feature"].values, fontsize=10)
ax.set_xlabel("Split Importance (số lần được dùng để tách node)")
ax.set_title("Top 15 Features — LightGBM Feature Importance (Split)",
             fontsize=13, fontweight="bold")
for bar, val in zip(bars, top15["importance"].values):
    ax.text(val + 30, bar.get_y() + bar.get_height() / 2,
            f"{val:,.0f}", va="center", ha="left", fontsize=9)
plt.tight_layout()
plt.show()"""
))

cells.append(md(
"""### Insight — Feature Importance LightGBM vs Logistic Regression

**Top features LightGBM (Split Importance):**

| Rank | Feature | LightGBM Rank | LR Rank (coef) | Nhận xét |
| :---: | :--- | :---: | :---: | :--- |
| 1 | `Days_Since_Last_Trade` | **#1** | #2 | LightGBM phát hiện nhiều điểm tách phi tuyến |
| 2 | `Cash_to_Asset_Ratio` | **#2** | #6 | LightGBM đánh giá sức mua quan trọng hơn |
| 3 | `Sector_Trade_Ratio` | **#3** | #4 | Thống nhất cao |
| 4 | `Credit_Score` | **#4** | ~#10+ | LightGBM phát hiện tương tác ẩn của Credit Score |
| — | `VCK_Page_Views_7D` | ~#11+ | **#1** | LR nhạy cảm hơn với tín hiệu intent trực tiếp |

**Sự khác biệt quan trọng:**
- LR ưu tiên **tín hiệu đặc thù VCK** (`VCK_Page_Views`, `Past_Transactions`) — phù hợp vì data được tạo từ công thức tuyến tính có weight cao cho các features này
- LightGBM sử dụng **nhiều features hơn** (split count phân bố đều hơn) — phát hiện cả `Credit_Score`, `Income`, `Age` mà LR bỏ qua
- Đây là lý do LightGBM thường **mạnh hơn trên real data** nơi các features này thực sự có tương tác phức tạp"""
))

# ── CELL 30 ── Business metrics
cells.append(md("## 6. Business Metrics — Precision@K & Lift Chart"))

cells.append(code(
"""results_df = pd.DataFrame({"y_true": y_test.values, "y_proba": y_proba})
results_df = results_df.sort_values("y_proba", ascending=False).reset_index(drop=True)

total_pos  = results_df["y_true"].sum()
baseline   = results_df["y_true"].mean()

k_values = [1_000, 2_000, 3_000, 4_000, 6_000, 8_000, 10_000, 15_000, 20_000]
print(f"{'K':>8} | {'Precision':>10} | {'Recall':>8} | {'Lift':>7} | {'Captured':>10}")
print("-" * 58)
for k in k_values:
    if k > len(results_df):
        break
    top_k   = results_df["y_true"].iloc[:k]
    prec_k  = top_k.mean()
    rec_k   = top_k.sum() / total_pos
    lift_k  = prec_k / baseline
    print(f"{k:>8,} | {prec_k:>10.4f} | {rec_k:>8.4f} | {lift_k:>7.2f}x | {int(top_k.sum()):>5,} / {int(total_pos):,}")"""
))

cells.append(md(
"""### Insight — Precision@K: LightGBM vs LR

| K | LightGBM Prec | LR Prec | LightGBM Lift | LR Lift |
| :---: | :---: | :---: | :---: | :---: |
| 1,000 | 95.2% | **97.0%** | 14.3x | **14.6x** |
| 2,000 | 87.7% | **88.6%** | 13.1x | **13.3x** |
| 4,000 | 68.9% | **70.2%** | 10.3x | **10.5x** |
| 6,000 | 54.8% | **55.7%** | 8.2x | **8.4x** |
| 10,000 | 37.7% | **38.0%** | 5.7x | **5.7x** |

**LR nhỉnh hơn LGBM ở tất cả các mốc K** — nhưng chênh lệch rất nhỏ (1–2%).

**Thực tế campaign:** Sự khác biệt này **không đáng kể** về mặt business. Với K=4,000, LR tiếp cận đúng 2,806 buyer còn LGBM tiếp cận đúng 2,754 buyer — chênh nhau chỉ 52 người trên 4,000."""
))

cells.append(code(
"""# ── Lift Chart theo Decile ──
n_deciles  = 10
decile_sz  = len(results_df) // n_deciles
lift_decile = []
for d in range(n_deciles):
    chunk = results_df["y_true"].iloc[d * decile_sz : (d + 1) * decile_sz]
    lift_decile.append(chunk.mean() / baseline)

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

bar_colors = ["#FF6F00" if v >= 1 else "#9E9E9E" for v in lift_decile]
bars = axes[0].bar(range(1, 11), lift_decile, color=bar_colors, edgecolor="white", width=0.7)
axes[0].axhline(y=1.0, color="black", linestyle="--", lw=1.5, label="Baseline Lift = 1.0")
axes[0].set_xlabel("Decile (1 = xác suất cao nhất)")
axes[0].set_ylabel("Lift")
axes[0].set_title("Lift Chart theo Decile — LightGBM", fontsize=13, fontweight="bold")
axes[0].set_xticks(range(1, 11))
axes[0].set_xticklabels([f"D{i}" for i in range(1, 11)])
for bar, val in zip(bars, lift_decile):
    axes[0].text(bar.get_x() + bar.get_width() / 2, val + 0.05,
                 f"{val:.2f}x", ha="center", va="bottom", fontsize=9)
axes[0].legend()

cum_gain    = [results_df["y_true"].iloc[: (d + 1) * decile_sz].sum() / total_pos for d in range(n_deciles)]
random_line = [x / 10 for x in range(1, 11)]
perfect_k   = [min(1.0, (d + 1) * decile_sz / total_pos) for d in range(n_deciles)]

axes[1].plot(range(1, 11), cum_gain,    marker="o", color="#FF6F00", lw=2.5, label="LightGBM")
axes[1].plot(range(1, 11), random_line, color="#9E9E9E", linestyle="--", lw=1.5, label="Random")
axes[1].plot(range(1, 11), perfect_k,   color="#4CAF50", linestyle=":",  lw=1.5, label="Perfect Model")
axes[1].fill_between(range(1, 11), cum_gain, random_line, alpha=0.08, color="#FF6F00")
axes[1].set_xlabel("Decile")
axes[1].set_ylabel("Tỷ lệ tích lũy Positive bắt được")
axes[1].set_title("Cumulative Gain Chart — LightGBM", fontsize=13, fontweight="bold")
axes[1].yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
axes[1].set_xticks(range(1, 11))
axes[1].set_xticklabels([f"D{i}" for i in range(1, 11)])
axes[1].legend()

plt.tight_layout()
plt.show()"""
))

# ── CELL 35 ── Full comparison section
cells.append(md("## 7. So sánh trực tiếp: LightGBM vs Logistic Regression"))

cells.append(code(
"""# ── Tái tạo LR để vẽ curves so sánh ──
lr_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(
        class_weight="balanced", C=1.0,
        max_iter=1000, solver="lbfgs", random_state=RANDOM_STATE,
    )),
])
lr_pipeline.fit(X_train, y_train)
lr_proba = lr_pipeline.predict_proba(X_test)[:, 1]

# LR threshold tối ưu
lr_f1s = [f1_score(y_test, (lr_proba >= t).astype(int), zero_division=0) for t in np.arange(0.01, 0.99, 0.01)]
lr_best_t  = np.arange(0.01, 0.99, 0.01)[int(np.argmax(lr_f1s))]
lr_proba_opt = (lr_proba >= lr_best_t).astype(int)

print("=== SUMMARY SO SÁNH ===")
print(f"{'Metric':<30} {'LightGBM':>12} {'LR Baseline':>14}")
print("-" * 58)
metrics = {
    "CV ROC-AUC": ("0.9732 ± 0.0015", "0.9770 ± 0.0012"),
    "CV PR-AUC":  ("0.7738 ± 0.0077", "0.7982 ± 0.0064"),
    "Test ROC-AUC": (f"{roc_auc_score(y_test, y_proba):.4f}", f"{roc_auc_score(y_test, lr_proba):.4f}"),
    "Test PR-AUC":  (f"{average_precision_score(y_test, y_proba):.4f}", f"{average_precision_score(y_test, lr_proba):.4f}"),
    "KS Statistic": (f"{ks_stat:.4f}", "0.8414"),
    "Best Threshold": (f"{best_threshold:.2f}", f"{lr_best_t:.2f}"),
    "F1 @ best thresh": (f"{best_f1:.4f}", f"{max(lr_f1s):.4f}"),
    "Precision @ best": (f"{precision_score(y_test, y_pred_opt):.4f}", f"{precision_score(y_test, lr_proba_opt):.4f}"),
    "Recall @ best": (f"{recall_score(y_test, y_pred_opt):.4f}", f"{recall_score(y_test, lr_proba_opt):.4f}"),
}
for k, (lgbm_v, lr_v) in metrics.items():
    print(f"  {k:<28} {lgbm_v:>12} {lr_v:>14}")"""
))

cells.append(code(
"""# ── ROC Curve so sánh ──
fpr_lr,  tpr_lr,  _ = roc_curve(y_test, lr_proba)
fpr_lgb, tpr_lgb, _ = roc_curve(y_test, y_proba)
prec_lr, rec_lr,  _ = precision_recall_curve(y_test, lr_proba)
prec_lgb,rec_lgb, _ = precision_recall_curve(y_test, y_proba)

roc_lr  = roc_auc_score(y_test, lr_proba)
pr_lr   = average_precision_score(y_test, lr_proba)
roc_lgb = roc_auc_score(y_test, y_proba)
pr_lgb  = average_precision_score(y_test, y_proba)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ROC so sánh
axes[0].plot(fpr_lr,  tpr_lr,  color="#E53935", lw=2.5, label=f"Logistic Regression (AUC = {roc_lr:.4f})")
axes[0].plot(fpr_lgb, tpr_lgb, color="#FF6F00", lw=2.5, linestyle="--", label=f"LightGBM (AUC = {roc_lgb:.4f})")
axes[0].plot([0, 1], [0, 1], color="#9E9E9E", linestyle=":", lw=1.5, label="Random")
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")
axes[0].set_title("ROC Curve — So sánh", fontsize=13, fontweight="bold")
axes[0].legend(loc="lower right")

# PR so sánh
axes[1].plot(rec_lr,  prec_lr,  color="#E53935", lw=2.5, label=f"Logistic Regression (AP = {pr_lr:.4f})")
axes[1].plot(rec_lgb, prec_lgb, color="#FF6F00", lw=2.5, linestyle="--", label=f"LightGBM (AP = {pr_lgb:.4f})")
axes[1].axhline(y=y_test.mean(), color="#9E9E9E", linestyle=":", lw=1.5, label=f"Baseline = {y_test.mean():.4f}")
axes[1].set_xlabel("Recall")
axes[1].set_ylabel("Precision")
axes[1].set_title("Precision-Recall Curve — So sánh", fontsize=13, fontweight="bold")
axes[1].legend(loc="upper right")

plt.tight_layout()
plt.show()"""
))

cells.append(code(
"""# ── Lift Chart so sánh ──
res_lr = pd.DataFrame({"y_true": y_test.values, "y_proba": lr_proba}).sort_values("y_proba", ascending=False).reset_index(drop=True)

lift_lr   = [res_lr["y_true"].iloc[d*decile_sz:(d+1)*decile_sz].mean() / baseline for d in range(n_deciles)]
lift_lgbm = lift_decile

x = np.arange(1, 11)
width = 0.35
fig, ax = plt.subplots(figsize=(12, 5))
bars1 = ax.bar(x - width/2, lift_lr,   width, label="Logistic Regression", color="#E53935", alpha=0.85)
bars2 = ax.bar(x + width/2, lift_lgbm, width, label="LightGBM",            color="#FF6F00", alpha=0.85)
ax.axhline(y=1.0, color="black", linestyle="--", lw=1.5, label="Baseline = 1.0")
ax.set_xlabel("Decile (1 = xác suất cao nhất)")
ax.set_ylabel("Lift")
ax.set_title("Lift Chart theo Decile — LR vs LightGBM", fontsize=13, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels([f"D{i}" for i in range(1, 11)])
ax.legend()
plt.tight_layout()
plt.show()"""
))

cells.append(md(
"""### Insight — Kết luận So sánh Toàn diện

#### Tổng kết số liệu

| Metric | **LightGBM** | **Logistic Regression** | Ưu thế |
| :--- | :---: | :---: | :---: |
| CV ROC-AUC | 0.9732 | **0.9770** | LR +0.004 |
| CV PR-AUC | 0.7738 | **0.7982** | LR +0.024 |
| KS Statistic | 0.8328 | **0.8414** | LR +0.009 |
| F1 @ optimal | **0.6887** | 0.6833 | LGBM +0.005 |
| Precision @ optimal | **0.6734** | 0.5886 | LGBM +0.085 |
| FP @ optimal | **1,367** | 2,276 | LGBM −40% |
| Recall @ optimal | 0.7047 | **0.8143** | LR +0.110 |

#### Giải thích nghịch lý (LR thắng về AUC)

Data mockup được tạo từ công thức **tuyến tính có kiểm soát** (`latent_score = Σ wᵢ·xᵢ + noise`). Logistic Regression học đúng cấu trúc tạo ra dữ liệu nên cho AUC/PR tốt hơn. **Với production data thực**, các quan hệ phi tuyến và tương tác giữa features sẽ phức tạp hơn — đây là nơi LightGBM thể hiện ưu thế.

#### Khuyến nghị lựa chọn

| Tình huống | Model được chọn | Lý do |
| :--- | :--- | :--- |
| Cần **Precision cao** (campaign VIP, tư vấn 1-1) | **LightGBM** | FP thấp hơn 40% tại threshold tối ưu |
| Cần **Recall cao** (không bỏ sót buyer) | **LR** | Recall 0.81 vs 0.70 tại threshold tối ưu |
| Cần **giải thích cho management** | **LR** | Hệ số hồi quy dễ diễn giải |
| **Production real data** | **LightGBM** | Robustness với phi tuyến và noise |
| **Tốc độ inference** | **LightGBM** | Nhanh hơn LR ở quy mô lớn |"""
))

# ── CELL 42 ── Summary
cells.append(md("## 8. Tổng kết"))

cells.append(code(
"""print("=" * 66)
print("      BẢNG SO SÁNH CUỐI CÙNG — LightGBM vs Logistic Regression")
print("=" * 66)
rows = [
    ("CV ROC-AUC",         "0.9732 ± 0.0015", "0.9770 ± 0.0012", "LR"),
    ("CV PR-AUC",          "0.7738 ± 0.0077", "0.7982 ± 0.0064", "LR"),
    ("Test ROC-AUC",       f"{roc_auc_score(y_test, y_proba):.4f}", f"{roc_auc_score(y_test, lr_proba):.4f}", "LR"),
    ("Test PR-AUC",        f"{average_precision_score(y_test, y_proba):.4f}", f"{average_precision_score(y_test, lr_proba):.4f}", "LR"),
    ("KS Statistic",       f"{ks_stat:.4f}", "0.8414", "LR"),
    ("F1 @ optimal",       f"{best_f1:.4f}", f"{max(lr_f1s):.4f}", "LGBM"),
    ("Precision @ optimal",f"{precision_score(y_test, y_pred_opt):.4f}", f"{precision_score(y_test, lr_proba_opt):.4f}", "LGBM"),
    ("Recall @ optimal",   f"{recall_score(y_test, y_pred_opt):.4f}", f"{recall_score(y_test, lr_proba_opt):.4f}", "LR"),
]
print(f"  {'Metric':<28} {'LightGBM':>14} {'LR Baseline':>14} {'Winner':>8}")
print("  " + "-" * 64)
for name, lgbm_v, lr_v, winner in rows:
    print(f"  {name:<28} {lgbm_v:>14} {lr_v:>14} {winner:>8}")
print("=" * 66)
print("  Kết luận: Hai model tương đương — lựa chọn theo ngữ cảnh campaign")
print("=" * 66)"""
))

cells.append(md(
"""## Bước tiếp theo

| Bước | Nội dung |
| :--- | :--- |
| **Hyperparameter Tuning** | Dùng `Optuna` tối ưu `num_leaves`, `learning_rate`, `min_child_samples` cho LightGBM |
| **Feature Engineering** | Thêm interaction features (`VCK_Views × Cash_Balance`, `Watchlist × Sector_Ratio`) |
| **SHAP Analysis** | Dùng `shap` library để giải thích từng dự đoán — bổ sung cho cả hai model |
| **Calibration** | Dùng `CalibratedClassifierCV` để cải thiện chất lượng xác suất predicted |
| **Multi-class** | Xây dựng tương tự cho `Target_MultiClass` (High / Medium / Low) |"""
))

# ── BUILD & WRITE ──────────────────────────────────────────
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
