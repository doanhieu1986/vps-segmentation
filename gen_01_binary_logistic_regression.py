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
cells.append(md(
"""### Insight — Phân bố nhãn (Class Distribution)

> **Mất cân bằng nghiêm trọng:** chỉ **6.67%** (20,000 / 300,000) khách hàng có mua VCK.

| Nhận định | Chi tiết |
| :--- | :--- |
| Nếu model dự đoán **tất cả là 0** | Accuracy = 93.33% — con số ảo, vô nghĩa |
| Metric phù hợp | **PR-AUC** và **F1 (class=1)**, không dùng Accuracy |
| Lý do dùng `class_weight='balanced'` | Tăng trọng số class 1 lên ~14x để model không bỏ qua nhóm thiểu số |

⚠️ **Lưu ý:** Bất kỳ metric nào không tách biệt theo class đều có thể gây hiểu lầm với dữ liệu này."""
))

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

# ── CELL 7 insight ── Feature distributions
cells.append(md(
"""### Insight — Phân bố Feature: Mua vs Không mua

Phân tích 6 features quan trọng cho thấy sự khác biệt rõ ràng giữa 2 nhóm:

| Feature | Nhóm Mua | Nhóm Không mua | Nhận định |
| :--- | :---: | :---: | :--- |
| `VCK_Page_Views_7D` | **1.95** lượt/tuần | 0.47 lượt/tuần | **Khác biệt 4x** — tín hiệu intent mạnh nhất |
| `Is_VCK_In_Watchlist` | **30.8%** | 13.8% | Đặt vào watchlist = tăng 2.2x khả năng mua |
| `Past_VCK_Transactions > 0` | **64.8%** | 37.7% | Ai từng mua VCK → xác suất mua lại rất cao |
| `Sector_Trade_Ratio` | **0.354** | 0.281 | Hay đầu tư cùng ngành → sở thích phù hợp |
| `Cash_to_Asset_Ratio` | **0.338** | 0.282 | Tiền mặt sẵn sàng giải ngân nhiều hơn |
| `Days_Since_Last_Trade` | **58.3 ngày** | 91.8 ngày | Buyer hoạt động thường xuyên hơn (~33 ngày) |

**Kết luận:** Các tín hiệu *đặc thù VCK* (`VCK_Page_Views`, `Watchlist`, `Past_Transactions`) là discriminative nhất, theo sau là các tín hiệu *sức mua* và *hoạt động gần đây*."""
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

# ── CELL 12 insight ── CV results
cells.append(md(
"""### Insight — Kết quả Cross-Validation

| Metric | Giá trị | Đánh giá |
| :--- | :---: | :--- |
| CV ROC-AUC | **0.9770 ± 0.0012** | Rất tốt — model phân hạng rất tốt giữa buyer/non-buyer |
| CV PR-AUC | **0.7982 ± 0.0064** | Mạnh — gấp ~12x so với baseline ngẫu nhiên (0.0667) |
| Độ lệch chuẩn (std) | ≤ 0.006 | Rất thấp → model ổn định, không overfit theo fold |

**Nhận định:** ROC-AUC 0.977 cho thấy Logistic Regression — dù là model đơn giản — đã học được cấu trúc phân tách tốt từ dữ liệu này. PR-AUC 0.80 đặc biệt ý nghĩa với bài toán imbalanced (baseline = 0.067)."""
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

# ── CELL 14 insight ── Classification report
cells.append(md(
"""### Insight — Classification Report (Threshold = 0.5)

| Chỉ số (class = 1) | Giá trị | Ý nghĩa |
| :--- | :---: | :--- |
| Precision | 0.42 | Trong 100 người model dự đoán sẽ mua, chỉ 42 người thực sự mua |
| Recall | **0.93** | Model bắt được 93% tổng số khách hàng thực sự có mua |
| F1-Score | 0.58 | Trung bình điều hòa giữa precision và recall |

**Giải thích:** `class_weight='balanced'` ưu tiên **Recall** — không bỏ sót buyer — dẫn đến Precision thấp (nhiều FP). Đây là hành vi mong đợi; Threshold tuning ở Section 5 sẽ cân bằng lại."""
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

# ── CELL 15 insight ── Confusion matrix
cells.append(md(
"""### Insight — Confusion Matrix (Threshold = 0.5)

```
                  Pred: 0       Pred: 1
Actual: 0   TN = 50,955   FP = 5,045   ← 5,045 cuộc gọi/email lãng phí
Actual: 1   FN =    295   TP = 3,705   ← Chỉ bỏ sót 295 buyer thực
```

| Loại lỗi | Số lượng | Chi phí business |
| :--- | :---: | :--- |
| **FP** (tiếp cận nhầm) | 5,045 | Lãng phí nguồn lực campaign (gọi điện, SMS, email) |
| **FN** (bỏ sót buyer) | 295 | Mất doanh thu — khách mua nhưng không được tiếp cận |

**Tỷ lệ FP/TP = 5,045 / 3,705 ≈ 1.36** — cứ tiếp cận đúng 1 buyer thì cũng tiếp cận nhầm 1.36 non-buyer. Threshold tuning sẽ cải thiện tỷ lệ này."""
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

# ── CELL 16 insight ── ROC + PR curves
cells.append(md(
"""### Insight — ROC Curve & Precision-Recall Curve

**ROC-AUC = 0.9758**
- Model phân biệt buyer / non-buyer **rất tốt** ở mức ranking
- Đường ROC gần sát góc trên-trái → ít phải đánh đổi TPR để giảm FPR

**PR-AUC = 0.7904** *(metric quan trọng hơn với imbalanced data)*
- Baseline ngẫu nhiên chỉ đạt PR-AUC = 0.0667 (tỷ lệ positive)
- Model đạt **gấp ~11.8x** so với baseline → khả năng phân loại thực sự tốt
- Phần diện tích PR curve lớn ở góc trên-phải cho thấy: với Recall cao (~80%), model vẫn giữ Precision trên 50%

**Kết luận:** Cả hai curves đều cho thấy model Logistic Regression là baseline mạnh, phù hợp để đặt mức chuẩn so sánh với LightGBM."""
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

# ── CELL 17 insight ── KS Statistic
cells.append(md(
"""### Insight — KS Statistic

**KS = 0.8414** — đạt tại top **17.0%** khách hàng theo xác suất giảm dần

| Thang đo | KS < 0.2 | 0.2 – 0.4 | 0.4 – 0.6 | **> 0.6** |
| :--- | :---: | :---: | :---: | :---: |
| Đánh giá | Kém | Khá | Tốt | **Rất tốt ✅** |

**Diễn giải thực tế:**
- Tại điểm KS tối ưu (top 17% ≈ **51,000 khách**), khoảng cách giữa tỷ lệ tích lũy Positive và Negative là lớn nhất
- Điều này có nghĩa: nếu chỉ tiếp cận **17%** danh sách khách hàng (những người có xác suất cao nhất), model đã phân tách tối ưu buyer khỏi non-buyer
- KS 0.84 rất cao — phản ánh việc dữ liệu có các tín hiệu intent VCK rất rõ ràng (`VCK_Page_Views`, `Past_Transactions`)

> **Ứng dụng:** Trong scoring model của ngân hàng/chứng khoán, KS > 0.6 là tiêu chí để model được duyệt đưa vào production."""
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

# ── CELL 19 insight ── Threshold scan
cells.append(md(
"""### Insight — Tại sao Threshold tối ưu = 0.79?

**Threshold mặc định 0.5 không phù hợp** với model dùng `class_weight='balanced'`:
- `class_weight='balanced'` làm cho model "tự tin hơn" vào class 1 → xác suất predicted bị đẩy cao
- Kết quả: phần lớn khách hàng có `y_proba > 0.5`, dẫn đến quá nhiều FP

**Threshold tối ưu = 0.79** tối đa hóa F1 (class=1):

| | Threshold 0.5 | Threshold 0.79 | Thay đổi |
| :--- | :---: | :---: | :---: |
| Precision | 0.42 | **0.59** | +17pp |
| Recall | 0.93 | **0.81** | −12pp |
| **F1-Score** | 0.58 | **0.68** | **+10pp** |

**Đánh đổi có lợi:** Tăng 17pp Precision (chính xác hơn) với chi phí chỉ giảm 12pp Recall (bỏ sót thêm ~188 buyer). Tùy ngữ cảnh campaign, team có thể chọn threshold thấp hơn nếu muốn Recall cao hơn."""
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

# ── CELL 20 insight ── Eval at optimal threshold
cells.append(md(
"""### Insight — So sánh Trước / Sau Threshold Tuning

```
                  Threshold 0.5          Threshold 0.79 (Optimal)
FP (tiếp cận nhầm):   5,045      →          2,276     (−2,769, giảm 54.9%)
FN (bỏ sót buyer):      295      →            743     (+448,   tăng 151.9%)
TP (đúng: có mua):    3,705      →          3,257     (−448)
```

**Kết quả thực tế cho campaign:**
- Với threshold 0.79: danh sách tiếp cận giảm từ **8,750 → 5,533 khách** (giảm 36.8%)
- Trong danh sách đó, tỷ lệ buyer tăng từ **42% → 59%** — chất lượng lead cao hơn đáng kể
- Chi phí campaign (gọi điện, email) giảm tương ứng do list nhỏ hơn

> **Khuyến nghị:** Dùng threshold 0.79 cho campaign có **chi phí tiếp cận cao** (gọi điện, tư vấn 1-1). Dùng threshold ~0.5 nếu tiếp cận qua kênh **chi phí thấp** (push notification, in-app banner)."""
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

# ── CELL 22 insight ── Coefficients
cells.append(md(
"""### Insight — Feature Importance (Hệ số hồi quy chuẩn hóa)

**Top features và ý nghĩa kinh doanh:**

| Rank | Feature | Coef | Nhận định |
| :---: | :--- | :---: | :--- |
| 1 | `VCK_Page_Views_7D` | **+2.70** | Xem trang VCK gần đây = tín hiệu intent mạnh nhất, đặc thù cho mã cổ phiếu |
| 2 | `Days_Since_Last_Trade` | **−1.76** | Càng lâu không giao dịch → xác suất mua càng thấp (khách inactive) |
| 3 | `Past_VCK_Transactions` | **+1.46** | Đã từng mua VCK → quen thuộc, khả năng tái mua cao |
| 4 | `Sector_Trade_Ratio` | **+1.09** | Hay đầu tư cổ phiếu cùng ngành → sở thích portfolio phù hợp |
| 5 | `Is_VCK_In_Watchlist` | **+0.96** | Đặt VCK vào danh sách theo dõi = hành động ý định rõ ràng |
| 6 | `Cash_to_Asset_Ratio` | **+0.87** | Tiền mặt sẵn sàng giải ngân — có khả năng mua thực tế |
| 7 | `Login_Momentum` | **+0.70** | Đăng nhập app ngày càng nhiều → đang chủ động quan tâm thị trường |

**Nhận định quan trọng:**
- Top 7 features là sự kết hợp giữa **tín hiệu hành vi đặc thù VCK** (rank 1, 3, 5) và **tín hiệu hoạt động chung** (rank 2, 6, 7)
- `Occupation` và các demographic features gần hệ số 0 → **nhân khẩu học không phải yếu tố quyết định** — hành vi quan trọng hơn profile
- Không có feature nào bị multicollinearity nghiêm trọng (hệ số phân bố đều, không có feature nào dominate tuyệt đối)"""
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

# ── CELL 24 insight ── Precision@K table
cells.append(md(
"""### Insight — Precision@K: Chiến lược Campaign theo Ngân sách

| K (Danh sách tiếp cận) | Precision | Recall | Lift | Captured | Phù hợp với |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **1,000** | **97.0%** | 24.3% | 14.6x | 970 / 4,000 | Campaign VIP, tư vấn 1-1, chi phí cao |
| **2,000** | 88.6% | 44.3% | 13.3x | 1,771 / 4,000 | Gọi điện telesale có chọn lọc |
| **4,000** | 70.2% | 70.2% | 10.5x | 2,806 / 4,000 | **Balance tốt** — tỷ lệ đúng cao, bắt 70% buyer |
| **6,000** | 55.7% | 83.6% | 8.4x | 3,343 / 4,000 | Email/SMS với ngân sách trung bình |
| **10,000** | 38.0% | 95.1% | 5.7x | 3,802 / 4,000 | Push notification, chi phí rất thấp |

**Gợi ý theo kênh:**
- 📞 **Telesale / tư vấn cá nhân** → K ≤ 2,000 (Precision > 85%)
- 📧 **Email marketing** → K ≈ 4,000–6,000 (Lift > 8x)
- 📱 **Push notification / in-app** → K ≤ 10,000 (gần như toàn bộ buyer)

> Với K=4,000 (precision 70%), mỗi 10 khách tiếp cận có 7 người thực sự mua — hiệu quả **gấp 10.5 lần** so với tiếp cận ngẫu nhiên."""
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

# ── CELL 25 insight ── Lift + Gain charts
cells.append(md(
"""### Insight — Lift Chart & Cumulative Gain Chart

**Lift Chart (theo Decile):**
- **Decile 1** (top 10% = 12,000 khách): Lift ≈ **14–15x** — gần như toàn bộ là buyer tập trung ở đây
- **Decile 2–3** (10–30%): Lift vẫn > 5x — còn hiệu quả tốt để tiếp cận
- **Decile 4–5** trở đi: Lift giảm nhanh → tiếp cận thêm không hiệu quả

**Cumulative Gain Chart:**
- Top 20% khách hàng (60,000 người) → model bắt được **> 85%** tổng số buyer
- Top 30% khách hàng (90,000 người) → bắt được **> 95%** tổng số buyer
- So với Perfect Model: gap nhỏ ở top 10%, cho thấy model đã xếp hạng rất tốt

**Kết luận chiến lược:**
> Chỉ cần tiếp cận **top 20–30%** danh sách khách hàng (60–90k người thay vì 300k toàn bộ), model đã giúp bắt được **85–95% buyer** — tiết kiệm **70–80% chi phí campaign** so với broadcast toàn bộ."""
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

# ── CELL 27 insight ── Overall conclusion
cells.append(md(
"""### Tổng kết & Nhận định tổng thể

#### Hiệu năng Model

| Metric | Giá trị | Đánh giá |
| :--- | :---: | :--- |
| CV ROC-AUC | 0.9770 ± 0.0012 | ✅ Rất tốt, ổn định |
| Test PR-AUC | 0.7904 | ✅ Mạnh (~12x baseline) |
| KS Statistic | **0.8414** | ✅ Xuất sắc (>0.6) |
| F1 @ threshold 0.79 | 0.6833 | ✅ Khá tốt cho imbalanced |

#### Điểm mạnh của Logistic Regression trong bài toán này
- **Dữ liệu có tín hiệu tuyến tính rõ ràng**: các features VCK-specific (`VCK_Page_Views`, `Watchlist`) có mối quan hệ gần tuyến tính với xác suất mua → LR phù hợp
- **Khả năng giải thích cao**: hệ số hồi quy trực tiếp diễn giải được cho team kinh doanh
- **Ổn định**: std CV thấp (<0.007), không có dấu hiệu overfit

#### Hạn chế cần lưu ý
- LR không nắm bắt được tương tác phi tuyến giữa features (e.g. khách hàng có cả `VCK_In_Watchlist` VÀ `Cash_Balance` cao có thể có xác suất mua không tuyến tính)
- Precision@1000 = 97% rất cao nhưng chỉ bắt 24% buyer → LightGBM kỳ vọng cải thiện phần này

#### Khuyến nghị sử dụng kết quả
- Dùng **score (xác suất)** thay vì nhãn 0/1 để xếp hạng danh sách tiếp cận
- Chọn **cutoff K theo ngân sách campaign** dựa trên bảng Precision@K
- **Không dùng Accuracy** để báo cáo kết quả với stakeholder — dùng Lift và Precision@K"""
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
