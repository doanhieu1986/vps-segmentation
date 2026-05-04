"""
Script sinh notebook: 01_binary_logistic_regression.ipynb
Chạy: python gen_01_binary_logistic_regression.py
"""
import json

OUTPUT = "01_binary_logistic_regression.ipynb"

def code(src): return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src}
def md(src):   return {"cell_type": "markdown", "metadata": {}, "source": src}

cells = []

# ── CELL 0 ── Title
cells.append(md(
"""# Binary Classification — Dự đoán mua cổ phiếu VCK
## Baseline Model: Logistic Regression

**Bài toán:** Dự đoán khách hàng có mua cổ phiếu VCK hay không (`Target_Binary`: 0 / 1)

**Pipeline:**
1. Load & Khám phá dữ liệu (EDA)
2. Tiền xử lý (Preprocessing)
3. Huấn luyện Logistic Regression với `class_weight='balanced'`
4. Đánh giá model (ROC-AUC, PR-AUC, KS Statistic)
5. Tối ưu Threshold
6. Feature Importance (hệ số hồi quy)
7. Business Metrics (Precision@K, Lift Chart)"""
))

# ── CELL 1 ── Imports
cells.append(code(
"""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
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
print("Libraries loaded successfully.")"""
))

# ── CELL 2 ── Section header
cells.append(md("## 1. Load & Khám phá dữ liệu"))

# ── CELL 3 ── Load data
cells.append(code(
"""df = pd.read_csv(DATA_PATH)
print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"\\nColumn dtypes:")
print(df.dtypes.to_string())
print(f"\\nMissing values: {df.isnull().sum().sum()}")"""
))

# ── CELL 4 ── Head
cells.append(code("df.head(5)"))

# ── CELL 5 ── Describe
cells.append(code(
"""# Thống kê mô tả cho các cột số
df.describe().round(2)"""
))

# ── CELL 6 ── Target distribution
cells.append(code(
"""# ── Phân bố Target Binary ──
print("=== PHÂN BỐ TARGET BINARY ===")
counts = df["Target_Binary"].value_counts().sort_index()
print(counts.rename({0: "Không mua (0)", 1: "Có mua (1)"}).to_string())
print(f"\\nTỷ lệ Positive (Có mua): {df['Target_Binary'].mean() * 100:.2f}%")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Bar chart
labels = ["Không mua (0)", "Có mua (1)"]
colors = ["#1976D2", "#E53935"]
bars = axes[0].bar(labels, counts.values, color=colors, width=0.5, edgecolor="white")
axes[0].set_title("Phân bố Target Binary", fontsize=13, fontweight="bold")
axes[0].set_ylabel("Số lượng khách hàng")
for bar, val in zip(bars, counts.values):
    axes[0].text(bar.get_x() + bar.get_width() / 2, val + 1500,
                 f"{val:,}\\n({val/len(df)*100:.1f}%)", ha="center", va="bottom", fontsize=11)
axes[0].set_ylim(0, max(counts.values) * 1.2)

# Pie chart
axes[1].pie(counts.values, labels=labels, autopct="%1.1f%%",
            colors=colors, startangle=90, explode=(0, 0.1),
            textprops={"fontsize": 11})
axes[1].set_title("Tỷ lệ Positive / Negative", fontsize=13, fontweight="bold")

plt.tight_layout()
plt.show()"""
))

# ── CELL 7 ── Feature distributions
cells.append(code(
"""# ── Phân bố 6 features quan trọng nhất ──
key_features = [
    "VCK_Page_Views_7D", "Is_VCK_In_Watchlist",
    "Cash_to_Asset_Ratio", "Sector_Trade_Ratio",
    "Login_Momentum", "Days_Since_Last_Trade",
]

fig, axes = plt.subplots(2, 3, figsize=(16, 8))
axes = axes.flatten()

for i, feat in enumerate(key_features):
    buyers     = df[df["Target_Binary"] == 1][feat]
    non_buyers = df[df["Target_Binary"] == 0][feat]
    axes[i].hist(non_buyers, bins=40, alpha=0.55, label="Không mua", color="#1976D2", density=True)
    axes[i].hist(buyers,     bins=40, alpha=0.70, label="Có mua",    color="#E53935", density=True)
    axes[i].set_title(feat, fontsize=11, fontweight="bold")
    axes[i].legend(fontsize=9)
    axes[i].set_yticks([])

plt.suptitle("Phân bố Feature: Mua vs Không mua VCK (density)", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.show()"""
))

# ── CELL 8 ── Preprocessing header
cells.append(md(
"""## 2. Tiền xử lý (Preprocessing)

| Feature | Xử lý |
| :--- | :--- |
| `Customer_ID` | Drop (không dùng trong model) |
| `Target_MultiClass` | Drop (không dùng trong bài toán này) |
| `Gender` | Label Encode: M → 1, F → 0 |
| `Occupation` | One-Hot Encoding (drop_first=True) |
| `City_Tier` | Giữ nguyên (ordinal numeric 1–3) |
| Tất cả numeric | StandardScaler (trong Pipeline) |"""
))

# ── CELL 9 ── Preprocessing code
cells.append(code(
"""df_model = df.drop(columns=["Customer_ID", "Target_MultiClass"]).copy()

# Encode Gender
df_model["Gender"] = (df_model["Gender"] == "M").astype(int)

# One-Hot Encode Occupation
df_model = pd.get_dummies(df_model, columns=["Occupation"], drop_first=True)

feature_cols = [c for c in df_model.columns if c != "Target_Binary"]
X = df_model[feature_cols]
y = df_model["Target_Binary"]

print(f"Số lượng features: {X.shape[1]}")
print(f"\\nDanh sách features:")
for i, col in enumerate(X.columns, 1):
    print(f"  {i:2d}. {col}")"""
))

# ── CELL 10 ── Train/test split
cells.append(code(
"""# ── Train / Test Split (stratified) ──
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

print(f"Train: {X_train.shape[0]:,} samples  |  Positive rate: {y_train.mean()*100:.2f}%")
print(f"Test:  {X_test.shape[0]:,} samples  |  Positive rate: {y_test.mean()*100:.2f}%")"""
))

# ── CELL 11 ── Training header
cells.append(md(
"""## 3. Huấn luyện Logistic Regression

**Chiến lược xử lý Class Imbalance:**
- `class_weight='balanced'` → tự động tăng trọng số cho class thiểu số (class=1)
- Xem xét điều chỉnh threshold (sẽ làm ở Section 5)

**Pipeline:** `StandardScaler` → `LogisticRegression`"""
))

# ── CELL 12 ── Train model
cells.append(code(
"""# ── Build Pipeline ──
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(
        class_weight="balanced",
        C=1.0,
        max_iter=1000,
        solver="lbfgs",
        random_state=RANDOM_STATE,
    )),
])

# ── Cross-Validation (5-fold, stratified) ──
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

print("=== CROSS-VALIDATION (5-fold, Stratified) ===")
cv_roc = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
cv_pr  = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="average_precision", n_jobs=-1)

print(f"  ROC-AUC : {cv_roc.mean():.4f} ± {cv_roc.std():.4f}  (per fold: {[round(v,4) for v in cv_roc]})")
print(f"  PR-AUC  : {cv_pr.mean():.4f} ± {cv_pr.std():.4f}  (per fold: {[round(v,4) for v in cv_pr]})")

# ── Train on full training set ──
pipeline.fit(X_train, y_train)
print("\\nModel đã train xong trên toàn bộ tập train.")"""
))

# ── CELL 13 ── Evaluation header
cells.append(md(
"""## 4. Đánh giá Model

> **Lưu ý:** Với class imbalance 6.67%, ưu tiên quan sát **PR-AUC** và **F1 (class=1)** hơn là Accuracy hay ROC-AUC đơn thuần."""
))

# ── CELL 14 ── Predictions + report
cells.append(code(
"""y_proba = pipeline.predict_proba(X_test)[:, 1]
y_pred  = pipeline.predict(X_test)  # default threshold = 0.5

print("=== ĐÁNH GIÁ TẠI THRESHOLD MẶC ĐỊNH (0.5) ===")
print(classification_report(y_test, y_pred, target_names=["Không mua (0)", "Có mua (1)"]))

roc_auc = roc_auc_score(y_test, y_proba)
pr_auc  = average_precision_score(y_test, y_proba)
print(f"ROC-AUC : {roc_auc:.4f}")
print(f"PR-AUC  : {pr_auc:.4f}")"""
))

# ── CELL 15 ── Confusion matrix
cells.append(code(
"""# ── Confusion Matrix ──
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, linewidths=0.5,
            xticklabels=["Pred: 0", "Pred: 1"],
            yticklabels=["Actual: 0", "Actual: 1"],
            annot_kws={"size": 14})
ax.set_title("Confusion Matrix  (Threshold = 0.5)", fontsize=13, fontweight="bold")
ax.set_ylabel("Actual", fontsize=12)
ax.set_xlabel("Predicted", fontsize=12)
plt.tight_layout()
plt.show()

print(f"TN (đúng: không mua)      : {tn:>8,}")
print(f"FP (sai: dự đoán mua)     : {fp:>8,}  ← Chi phí tiếp cận sai")
print(f"FN (sai: bỏ sót có mua)   : {fn:>8,}  ← Cơ hội bị bỏ lỡ")
print(f"TP (đúng: có mua)         : {tp:>8,}")"""
))

# ── CELL 16 ── ROC + PR curves
cells.append(code(
"""# ── ROC Curve + PR Curve ──
fpr, tpr, _ = roc_curve(y_test, y_proba)
precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_proba)
baseline_rate = y_test.mean()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ROC
axes[0].plot(fpr, tpr, color="#E53935", lw=2.5, label=f"Logistic Regression (AUC = {roc_auc:.4f})")
axes[0].plot([0, 1], [0, 1], color="#9E9E9E", linestyle="--", lw=1.5, label="Random Classifier")
axes[0].fill_between(fpr, tpr, alpha=0.07, color="#E53935")
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")
axes[0].set_title("ROC Curve", fontsize=13, fontweight="bold")
axes[0].legend(loc="lower right")

# PR
axes[1].plot(recall_vals, precision_vals, color="#1976D2", lw=2.5, label=f"Logistic Regression (AP = {pr_auc:.4f})")
axes[1].axhline(y=baseline_rate, color="#9E9E9E", linestyle="--", lw=1.5,
                label=f"Random baseline = {baseline_rate:.4f}")
axes[1].fill_between(recall_vals, precision_vals, alpha=0.07, color="#1976D2")
axes[1].set_xlabel("Recall")
axes[1].set_ylabel("Precision")
axes[1].set_title("Precision-Recall Curve", fontsize=13, fontweight="bold")
axes[1].legend(loc="upper right")

plt.tight_layout()
plt.show()"""
))

# ── CELL 17 ── KS Statistic
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
plt.axvline(x=ks_pct / 100, color="#4CAF50", linestyle="--", lw=1.8, label=f"KS = {ks_stat:.4f} (top {ks_pct:.1f}%)")
plt.fill_between(x_axis, ks_df["cum_pos_rate"].values, ks_df["cum_neg_rate"].values,
                 alpha=0.08, color="#4CAF50")
plt.xlabel("Tỷ lệ khách hàng (sắp xếp theo xác suất giảm dần)")
plt.ylabel("Tỷ lệ tích lũy")
plt.title("KS Statistic Chart", fontsize=13, fontweight="bold")
plt.legend()
plt.tight_layout()
plt.show()"""
))

# ── CELL 18 ── Threshold header
cells.append(md(
"""## 5. Tối ưu Threshold

Với class imbalance cao, threshold mặc định 0.5 thường không tối ưu.
Ta quét threshold và chọn giá trị tối ưu **F1-Score trên class=1**.

> Tùy yêu cầu business, có thể chọn threshold theo **Precision** (ưu tiên đúng hơn) hoặc **Recall** (ưu tiên không bỏ sót)."""
))

# ── CELL 19 ── Threshold scan
cells.append(code(
"""thresholds       = np.arange(0.05, 0.80, 0.01)
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
ax.axvline(x=best_threshold, color="#FF9800", linestyle="--", lw=2,
           label=f"Optimal threshold = {best_threshold:.2f}")
ax.scatter([best_threshold], [best_f1], color="#FF9800", s=100, zorder=5)
ax.set_xlabel("Threshold")
ax.set_ylabel("Score")
ax.set_title("Precision / Recall / F1 theo Threshold (class = 1)", fontsize=13, fontweight="bold")
ax.legend()
plt.tight_layout()
plt.show()"""
))

# ── CELL 20 ── Eval at optimal threshold
cells.append(code(
"""# ── Đánh giá tại Threshold tối ưu ──
y_pred_opt = (y_proba >= best_threshold).astype(int)

print(f"=== ĐÁNH GIÁ TẠI THRESHOLD TỐI ƯU ({best_threshold:.2f}) ===")
print(classification_report(y_test, y_pred_opt, target_names=["Không mua (0)", "Có mua (1)"]))

# So sánh trước / sau threshold tuning
cm_opt = confusion_matrix(y_test, y_pred_opt)
tn_o, fp_o, fn_o, tp_o = cm_opt.ravel()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, matrix, title in zip(
    axes,
    [cm, cm_opt],
    ["Confusion Matrix — Threshold 0.5 (Default)", f"Confusion Matrix — Threshold {best_threshold:.2f} (Optimal)"]
):
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", ax=ax, linewidths=0.5,
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

# ── CELL 21 ── Feature importance header
cells.append(md(
"""## 6. Feature Importance — Hệ số Logistic Regression

Magnitude của hệ số (sau khi StandardScaler) cho biết mức độ ảnh hưởng của từng feature.
- **Hệ số dương (đỏ):** Feature tăng → Xác suất mua tăng
- **Hệ số âm (xanh):** Feature tăng → Xác suất mua giảm"""
))

# ── CELL 22 ── Coefficients
cells.append(code(
"""clf   = pipeline.named_steps["clf"]
coefs = clf.coef_[0]

coef_df = (
    pd.DataFrame({"feature": X.columns, "coefficient": coefs})
    .assign(abs_coef=lambda d: d["coefficient"].abs())
    .sort_values("abs_coef", ascending=False)
    .reset_index(drop=True)
)

print("=== TOP 15 FEATURES THEO HỆ SỐ ===")
print(coef_df[["feature", "coefficient"]].head(15).to_string(index=False))

top15 = coef_df.head(15).sort_values("coefficient")
colors_bar = ["#E53935" if c > 0 else "#1976D2" for c in top15["coefficient"]]

fig, ax = plt.subplots(figsize=(10, 7))
bars = ax.barh(range(len(top15)), top15["coefficient"].values, color=colors_bar, edgecolor="white")
ax.set_yticks(range(len(top15)))
ax.set_yticklabels(top15["feature"].values, fontsize=10)
ax.axvline(x=0, color="black", linewidth=0.8)
ax.set_xlabel("Hệ số (Standardized Coefficient)")
ax.set_title("Top 15 Features — Logistic Regression Coefficients\\n(Đỏ = tăng P(mua), Xanh = giảm P(mua))",
             fontsize=13, fontweight="bold")
for bar, val in zip(bars, top15["coefficient"].values):
    ax.text(val + (0.01 if val >= 0 else -0.01), bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", va="center", ha="left" if val >= 0 else "right", fontsize=9)
plt.tight_layout()
plt.show()"""
))

# ── CELL 23 ── Business metrics header
cells.append(md(
"""## 7. Business Metrics — Precision@K & Lift Chart

Thực tế, campaign marketing/sales sẽ nhắm đến **Top-K khách hàng** có xác suất mua cao nhất.
- **Precision@K:** Trong K khách được chọn, có bao nhiêu % thực sự mua?
- **Lift@K:** Gấp bao nhiêu lần so với gọi ngẫu nhiên?
- **Recall@K:** Bắt được bao nhiêu % tổng số khách sẽ mua?"""
))

# ── CELL 24 ── Precision@K table
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

# ── CELL 25 ── Lift + Gain charts
cells.append(code(
"""# ── Lift Chart theo Decile ──
n_deciles  = 10
decile_sz  = len(results_df) // n_deciles
lift_decile = []
for d in range(n_deciles):
    chunk = results_df["y_true"].iloc[d * decile_sz : (d + 1) * decile_sz]
    lift_decile.append(chunk.mean() / baseline)

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Bar chart — Lift per decile
bar_colors = ["#E53935" if v >= 1 else "#9E9E9E" for v in lift_decile]
bars = axes[0].bar(range(1, 11), lift_decile, color=bar_colors, edgecolor="white", width=0.7)
axes[0].axhline(y=1.0, color="black", linestyle="--", lw=1.5, label="Baseline Lift = 1.0")
axes[0].set_xlabel("Decile (1 = xác suất cao nhất)")
axes[0].set_ylabel("Lift")
axes[0].set_title("Lift Chart theo Decile", fontsize=13, fontweight="bold")
axes[0].set_xticks(range(1, 11))
axes[0].set_xticklabels([f"D{i}" for i in range(1, 11)])
for bar, val in zip(bars, lift_decile):
    axes[0].text(bar.get_x() + bar.get_width() / 2, val + 0.05,
                 f"{val:.2f}x", ha="center", va="bottom", fontsize=9)
axes[0].legend()

# Cumulative Gain Chart
cum_gain    = [results_df["y_true"].iloc[: (d + 1) * decile_sz].sum() / total_pos for d in range(n_deciles)]
random_line = [x / 10 for x in range(1, 11)]
perfect_k   = [min(1.0, (d + 1) * decile_sz / total_pos) for d in range(n_deciles)]

axes[1].plot(range(1, 11), cum_gain,    marker="o", color="#E53935", lw=2.5, label="Model")
axes[1].plot(range(1, 11), random_line, color="#9E9E9E", linestyle="--", lw=1.5, label="Random")
axes[1].plot(range(1, 11), perfect_k,   color="#4CAF50", linestyle=":",  lw=1.5, label="Perfect Model")
axes[1].fill_between(range(1, 11), cum_gain, random_line, alpha=0.08, color="#E53935")
axes[1].set_xlabel("Decile")
axes[1].set_ylabel("Tỷ lệ tích lũy Positive bắt được")
axes[1].set_title("Cumulative Gain Chart", fontsize=13, fontweight="bold")
axes[1].yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
axes[1].set_xticks(range(1, 11))
axes[1].set_xticklabels([f"D{i}" for i in range(1, 11)])
axes[1].legend()

plt.tight_layout()
plt.show()"""
))

# ── CELL 26 ── Summary header
cells.append(md("## 8. Tổng kết kết quả"))

# ── CELL 27 ── Summary
cells.append(code(
"""roc_test = roc_auc_score(y_test, y_proba)
pr_test  = average_precision_score(y_test, y_proba)
f1_opt   = f1_score(y_test, y_pred_opt)
prec_opt = precision_score(y_test, y_pred_opt)
rec_opt  = recall_score(y_test, y_pred_opt)

summary = {
    "Model":               "Logistic Regression (class_weight=balanced)",
    "Train size":          f"{X_train.shape[0]:,}",
    "Test size":           f"{X_test.shape[0]:,}",
    "Số features":         X_train.shape[1],
    "CV ROC-AUC":          f"{cv_roc.mean():.4f} ± {cv_roc.std():.4f}",
    "CV PR-AUC":           f"{cv_pr.mean():.4f} ± {cv_pr.std():.4f}",
    "Test ROC-AUC":        f"{roc_test:.4f}",
    "Test PR-AUC":         f"{pr_test:.4f}",
    "KS Statistic":        f"{ks_stat:.4f}  (top {ks_pct:.1f}%)",
    "Best Threshold":      f"{best_threshold:.2f}",
    "F1 @ best threshold": f"{f1_opt:.4f}",
    "Precision @ best":    f"{prec_opt:.4f}",
    "Recall @ best":       f"{rec_opt:.4f}",
}

print("=" * 62)
print("    TỔNG KẾT — LOGISTIC REGRESSION BASELINE")
print("=" * 62)
for k, v in summary.items():
    print(f"  {k:<28}: {v}")
print("=" * 62)"""
))

# ── CELL 28 ── Next steps
cells.append(md(
"""## Bước tiếp theo

| Bước | Nội dung |
| :--- | :--- |
| **Model so sánh** | Thay `LogisticRegression` bằng `LightGBMClassifier` để benchmark hiệu năng |
| **Feature Engineering** | Tạo thêm interaction features (e.g. `VCK_Views × Cash_Balance`) |
| **Hyperparameter Tuning** | Dùng `Optuna` hoặc `GridSearchCV` để tối ưu `C` |
| **SHAP Analysis** | Dùng `shap` library để giải thích từng dự đoán cụ thể |
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
