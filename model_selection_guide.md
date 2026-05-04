# Hướng dẫn lựa chọn Model - Bài toán Dự đoán khách hàng mua cổ phiếu VCK

## Tổng quan bài toán

| Đặc điểm | Chi tiết |
| :--- | :--- |
| Quy mô | 300,000 khách hàng |
| Class Imbalance (Binary) | ~6.67% Positive (Mua) / ~93.33% Negative (Không mua) |
| Phân bố (Multi-class) | High ~8.3% / Medium ~20% / Low ~71.7% |
| Loại dữ liệu | Tabular (mixed: numerical + categorical) |
| Đặc trưng nổi bật | Có các feature đặc thù cho VCK (`Is_VCK_In_Watchlist`, `VCK_Page_Views`, `Past_VCK_Transactions`) |

---

## Phần 1: Binary Classification (Nhãn 0 / 1)

### Lưu ý quan trọng về Class Imbalance

Với tỷ lệ positive chỉ **6.67%**, các kỹ thuật xử lý mất cân bằng là **bắt buộc**:

| Kỹ thuật | Mô tả | Khuyến nghị |
| :--- | :--- | :--- |
| `class_weight='balanced'` | Tự động điều chỉnh trọng số loss theo tỷ lệ lớp | Đơn giản, áp dụng hầu hết mọi model |
| `scale_pos_weight` (XGBoost/LightGBM) | Đặt ≈ 93/6.67 ≈ 14 | Hiệu quả cao với Gradient Boosting |
| SMOTE / ADASYN | Oversampling lớp thiểu số (minority class) | Dùng khi model vẫn thiên vị sau weight |
| Threshold tuning | Hạ ngưỡng phân loại từ 0.5 xuống (e.g. 0.3) | Luôn thực hiện sau khi train |

**Metric đánh giá ưu tiên:** `PR-AUC` (Precision-Recall AUC) > `F1-Score (class=1)` > `ROC-AUC`  
→ Tránh dùng `Accuracy` vì model dự đoán tất cả là 0 vẫn đạt 93.33%.

---

### 1.1 Logistic Regression

**Mô tả:** Mô hình tuyến tính cơ bản, dự đoán xác suất qua hàm sigmoid.

| | Chi tiết |
| :--- | :--- |
| **Ưu điểm** | - Đơn giản, huấn luyện rất nhanh trên 300k rows <br> - Xuất ra xác suất (probability) tự nhiên <br> - Dễ giải thích hệ số (coefficient) cho business <br> - Ổn định, ít overfit |
| **Nhược điểm** | - Giả định mối quan hệ tuyến tính giữa feature và log-odds <br> - Không nắm bắt được tương tác phi tuyến giữa các feature <br> - Cần encode cẩn thận cho categorical features <br> - Hiệu năng thường kém hơn các model phức tạp |
| **Khuyến nghị dùng** | **Baseline model** bắt buộc phải có để so sánh; khi stakeholder yêu cầu model đơn giản, minh bạch (Explainability > Performance) |

```python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(class_weight='balanced', C=1.0, max_iter=1000)
```

---

### 1.2 Random Forest

**Mô tả:** Ensemble của nhiều Decision Tree, dự đoán bằng majority vote.

| | Chi tiết |
| :--- | :--- |
| **Ưu điểm** | - Nắm bắt quan hệ phi tuyến và tương tác feature tốt <br> - Robust với outliers và giá trị thiếu <br> - Cung cấp `feature_importances_` dễ hiểu <br> - Ít cần tuning hyperparameter hơn Gradient Boosting |
| **Nhược điểm** | - Chậm hơn LightGBM khi dữ liệu lớn (300k × 25 features) <br> - Tốn nhiều RAM (lưu toàn bộ cây) <br> - Kém hơn Gradient Boosting trong nhiều benchmark thực tế <br> - Dễ overfit với `max_depth` sâu nếu không điều chỉnh |
| **Khuyến nghị dùng** | Khi cần model dễ tune, cần feature importance trực quan; dùng làm model thứ 2 sau LightGBM để so sánh |

```python
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=300, class_weight='balanced',
                               max_depth=15, n_jobs=-1, random_state=42)
```

---

### 1.3 XGBoost

**Mô tả:** Gradient Boosting Trees, xây dựng cây tuần tự để giảm residual error.

| | Chi tiết |
| :--- | :--- |
| **Ưu điểm** | - Hiệu năng cao trên dữ liệu tabular <br> - Xử lý tốt missing values tự động <br> - `scale_pos_weight` hiệu quả cho imbalanced data <br> - Regularization (L1/L2) tích hợp sẵn <br> - Hỗ trợ tốt trên GPU |
| **Nhược điểm** | - Chậm hơn LightGBM khi train (level-wise tree growth) <br> - Nhiều hyperparameter cần tuning <br> - Tốn nhiều memory hơn LightGBM |
| **Khuyến nghị dùng** | Khi có GPU; khi cần kết quả ổn định để production; khi đã quen dùng XGBoost ecosystem |

```python
import xgboost as xgb
model = xgb.XGBClassifier(scale_pos_weight=14, n_estimators=500,
                           max_depth=6, learning_rate=0.05,
                           subsample=0.8, eval_metric='aucpr')
```

---

### 1.4 LightGBM ⭐ (Khuyến nghị chính)

**Mô tả:** Gradient Boosting với leaf-wise tree growth, tối ưu cho dữ liệu lớn.

| | Chi tiết |
| :--- | :--- |
| **Ưu điểm** | - **Nhanh nhất** trong các Gradient Boosting (leaf-wise, histogram-based) <br> - Tiêu thụ ít memory hơn XGBoost <br> - Xử lý tốt categorical features native (`cat_feature`) <br> - `is_unbalance=True` hoặc `scale_pos_weight` đều hỗ trợ <br> - Hiệu năng tốt nhất trên tabular data (benchmark Kaggle, fintech) |
| **Nhược điểm** | - Dễ overfit hơn XGBoost nếu không regularize (leaf-wise growth) <br> - Cần cẩn thận với `num_leaves` và `min_data_in_leaf` <br> - Ít phổ biến trong môi trường enterprise hơn XGBoost |
| **Khuyến nghị dùng** | **Model chính cho production** — tốt nhất về trade-off tốc độ/hiệu năng; phù hợp 300k rows với mixed features |

```python
import lightgbm as lgb
model = lgb.LGBMClassifier(is_unbalance=True, n_estimators=1000,
                            num_leaves=63, learning_rate=0.05,
                            min_child_samples=50, subsample=0.8,
                            metric='average_precision')
```

---

### 1.5 CatBoost

**Mô tả:** Gradient Boosting của Yandex, tối ưu cho categorical features.

| | Chi tiết |
| :--- | :--- |
| **Ưu điểm** | - Xử lý categorical features tốt nhất (ordered target encoding nội bộ) <br> - Ít cần preprocessing cho categorical <br> - Ít overfit hơn XGBoost/LightGBM <br> - `auto_class_weights='Balanced'` hỗ trợ imbalance |
| **Nhược điểm** | - Chậm hơn LightGBM khi train <br> - Lợi thế categorical không quá lớn khi dataset chỉ có ~4 categorical features <br> - Ít tài liệu và cộng đồng hơn |
| **Khuyến nghị dùng** | Khi dataset có nhiều categorical features hơn hiện tại; hoặc dùng trong ensemble |

---

### 1.6 Multi-Layer Perceptron (Neural Network / MLP)

**Mô tả:** Mạng neural cơ bản với các fully-connected layers.

| | Chi tiết |
| :--- | :--- |
| **Ưu điểm** | - Nắm bắt các pattern phi tuyến phức tạp <br> - Linh hoạt với kiến trúc (dropout, batch norm) <br> - Dễ scale khi có thêm data |
| **Nhược điểm** | - **Không phù hợp cho bài toán này**: tabular data với 25 features, Gradient Boosting luôn thắng MLP <br> - Không interpretable <br> - Cần chuẩn hóa tất cả features (StandardScaler) <br> - Overfit dễ dàng với imbalanced data |
| **Khuyến nghị dùng** | **Không khuyến nghị** cho bài toán này. Chỉ thử nghiệm nếu đã tối ưu hết Gradient Boosting và muốn khám phá |

---

## Phần 2: Multi-class Classification (High / Medium / Low)

### Lưu ý đặc biệt

Nhãn `High > Medium > Low` có **thứ tự tự nhiên (ordinal)** — đây là điểm quan trọng khi chọn model và thiết kế chiến lược.

**Phân bố lớp:**
```
Low:    ~215,000 (71.7%) ← Dominant class
Medium:  ~60,000 (20.0%)
High:    ~25,000 ( 8.3%)
```

**Metric đánh giá:** `Weighted F1-Score`, `Macro F1-Score`, `Confusion Matrix`

---

### Chiến lược 1: Direct Multi-class Classification (Đơn giản)

Huấn luyện trực tiếp với 3 nhãn, để model tự xử lý.

**Áp dụng được với:** LightGBM, XGBoost, Random Forest, Logistic Regression

| | Chi tiết |
| :--- | :--- |
| **Ưu điểm** | - Đơn giản, một pipeline duy nhất <br> - Gradient Boosting xử lý multi-class tốt (softmax objective) <br> - Dễ maintain và deploy |
| **Nhược điểm** | - Không tận dụng được thứ tự ordinal (High > Medium > Low) <br> - Model có thể nhầm High → Low (lỗi nghiêm trọng về business) |
| **Khuyến nghị** | Dùng làm **baseline** cho Multi-class; kết hợp với class_weight để cân bằng |

```python
# LightGBM multi-class
model = lgb.LGBMClassifier(objective='multiclass', num_class=3,
                            class_weight='balanced', n_estimators=1000)
```

---

### Chiến lược 2: Ordinal Classification ⭐ (Khuyến nghị chính)

Tận dụng thứ tự `Low < Medium < High` bằng cách chuyển thành bài toán nhị phân tuần tự.

**Nguyên lý:** Xây dựng K-1 = 2 binary classifier:
- **Classifier 1:** P(score ≥ Medium) = P(Medium + High)
- **Classifier 2:** P(score ≥ High) = P(High)

Dự đoán cuối: `Low` nếu cả 2 đều = 0; `Medium` nếu Clf1=1, Clf2=0; `High` nếu cả 2 đều = 1.

| | Chi tiết |
| :--- | :--- |
| **Ưu điểm** | - Phù hợp nhất về mặt toán học với nhãn có thứ tự <br> - Giảm thiểu lỗi nhảy cấp (High → Low) <br> - Mỗi classifier có thể tối ưu riêng cho ngưỡng phù hợp |
| **Nhược điểm** | - Phức tạp hơn: cần train 2 model riêng biệt <br> - Deploy phức tạp hơn (2 pipeline) <br> - Cần đảm bảo P(≥High) ≤ P(≥Medium) (isotonic constraint) |
| **Khuyến nghị** | Dùng khi **lỗi nhảy cấp có chi phí cao** (e.g. gọi điện cho khách Low thay vì High) |

```python
from mord import LogisticAT  # Ordinal Logistic
# Hoặc tự implement với 2 LightGBM binary classifier
```

---

### Chiến lược 3: Two-Stage Cascade

**Stage 1:** Phân loại nhị phân (Low vs Non-Low)  
**Stage 2:** Trong Non-Low, phân loại (Medium vs High)

| | Chi tiết |
| :--- | :--- |
| **Ưu điểm** | - Tận dụng được model binary đã xây dựng ở Phần 1 <br> - Stage 1 tập trung lọc Low (dễ, dữ liệu nhiều) <br> - Stage 2 tập trung phân biệt Medium/High (khó, dữ liệu ít) <br> - Dễ giải thích cho business |
| **Nhược điểm** | - Lỗi Stage 1 lan truyền vào Stage 2 (error propagation) <br> - Stage 2 train trên tập nhỏ (85k khách), dễ overfit <br> - 2 model cần maintain |
| **Khuyến nghị** | Dùng khi **logic business rõ ràng**: trước hết lọc Low, sau đó ưu tiên hóa High trong pool tiềm năng |

---

## Phần 3: Bảng so sánh tổng hợp

### Binary Classification

| Model | Hiệu năng | Tốc độ Train | Interpretable | Xử lý Imbalance | Khuyến nghị |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Logistic Regression | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | class_weight | Baseline |
| Random Forest | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | class_weight | So sánh |
| XGBoost | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | scale_pos_weight | Production |
| **LightGBM** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | is_unbalance | **Chính** ✅ |
| CatBoost | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | auto_class_weights | Ensemble |
| MLP | ⭐⭐⭐ | ⭐⭐ | ⭐ | class_weight | Không khuyến nghị |

### Multi-class Classification

| Chiến lược | Độ phức tạp | Phù hợp thứ tự | Business Logic | Khuyến nghị |
| :--- | :---: | :---: | :---: | :---: |
| Direct Multi-class | ⭐ | ⭐⭐⭐ | ⭐⭐⭐ | Baseline |
| **Ordinal Classification** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | **Chính** ✅ |
| Two-Stage Cascade | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Khi cần giải thích |

---

## Phần 4: Quy trình được khuyến nghị

### Bước 1: Preprocessing
```
Numerical features  → StandardScaler (cho Logistic, MLP) hoặc để nguyên (LightGBM)
Categorical features → LabelEncoder hoặc One-Hot (Gender, Occupation, City_Tier)
Derived features     → Đã có sẵn: Login_Momentum, Cash_to_Asset_Ratio, Sector_Trade_Ratio
```

### Bước 2: Train Baseline
```
Logistic Regression (class_weight='balanced') → Ghi lại PR-AUC, F1
```

### Bước 3: Train Main Model
```
LightGBM với is_unbalance=True → Cross-validation 5-fold → Tối ưu threshold
```

### Bước 4: Threshold Tuning (Binary)
```python
from sklearn.metrics import precision_recall_curve
precision, recall, thresholds = precision_recall_curve(y_val, y_proba)
# Chọn threshold tối ưu F1 hoặc theo yêu cầu business (precision vs recall trade-off)
```

### Bước 5: Đánh giá
```
Binary:     PR-AUC, F1@threshold, Precision@K (Top 20k), KS-Statistic, Lift Chart
Multi-class: Weighted F1, Macro F1, Confusion Matrix, Classification Report
```

---

## Phần 5: Khuyến nghị theo tình huống

| Tình huống | Model được khuyến nghị |
| :--- | :--- |
| Cần nhanh, đơn giản, explainable cho management | Logistic Regression + Feature importance |
| Tối ưu hiệu năng cho production pipeline | LightGBM (Binary) + Direct Multi-class LightGBM |
| Muốn tối thiểu hóa lỗi nhảy cấp (High → Low) | LightGBM + Ordinal Cascade |
| Cần giải thích lý do từng dự đoán (SHAP) | LightGBM + `shap` library |
| Muốn ensemble để tăng stability | LightGBM + XGBoost + Logistic (weighted voting) |
| Resource/compute hạn chế | LightGBM (nhanh nhất, ít RAM nhất) |

---

## Phần 6: Features quan trọng dự kiến

Dựa vào đặc trưng bài toán, các feature sau dự kiến có **predictive power cao nhất**:

| Nhóm | Feature | Lý do |
| :--- | :--- | :--- |
| **VCK-specific** | `Is_VCK_In_Watchlist`, `VCK_Page_Views_30D`, `Past_VCK_Transactions` | Tín hiệu intent trực tiếp nhất |
| **Trading Activity** | `Transaction_Count_3M`, `Days_Since_Last_Trade` | Đo lường khách hàng đang hoạt động |
| **App Engagement** | `Login_Momentum`, `App_Login_Freq_7D` | Tín hiệu hành vi gần nhất |
| **Financial Capacity** | `Cash_Balance_Mil`, `Cash_to_Asset_Ratio` | Khả năng mua thực tế |
| **Sector Affinity** | `Sector_Trade_Ratio` | Sở thích cổ phiếu cùng ngành VCK |

---

*Tài liệu này được tạo dựa trên phân tích đặc trưng dữ liệu trong `data_dictionary.md`. Cập nhật lần cuối: 2026-05-04.*
