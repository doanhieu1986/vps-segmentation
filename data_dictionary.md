# Data Dictionary - Bộ dữ liệu dự đoán hành vi mua cổ phiếu VCK (Propensity Model)

## 1. Thông tin chung
- **Quy mô dữ liệu (Dataset Size):** 300,000 dòng (Khách hàng)
- **Tỷ lệ nhãn (Class Imbalance):** Mất cân bằng với tỷ lệ ~6.67% Positive (Mua) và ~93.33% Negative (Không mua).
- **Mục tiêu (Objective):** Xây dựng mô hình phân loại (Classification Model) để dự đoán xác suất khách hàng sẽ mua mã cổ phiếu VCK.

---

## 2. Chi tiết các biến (Features)

### 🔑 Nhóm định danh (Identification)
| Tên cột (Feature) | Kiểu dữ liệu | Mô tả | Giá trị có thể có / Ví dụ |
| :--- | :--- | :--- | :--- |
| **Customer_ID** | String | Mã định danh duy nhất của từng khách hàng | `CUST_000001`, `CUST_000002` |

### 👤 Nhóm Nhân khẩu học (Demographic)
| Tên cột (Feature) | Kiểu dữ liệu | Mô tả | Giá trị có thể có / Ví dụ |
| :--- | :--- | :--- | :--- |
| **Age** | Integer | Tuổi của khách hàng | `18` đến `75` |
| **Gender** | Category | Giới tính | `M` (Nam), `F` (Nữ) |
| **City_Tier** | Integer | Phân loại thành phố sinh sống | `1`: HN/HCM, `2`: Tỉnh lớn, `3`: Tỉnh thành khác |
| **Income_Monthly_Mil** | Float | Thu nhập trung bình hàng tháng (Triệu VNĐ) | `15.5`, `30.0` |
| **Occupation** | Category | Nghề nghiệp hiện tại | `Office`, `Business`, `Freelance`, `Student/Retired` |

### 💰 Nhóm Tài sản & Tài chính (Asset / Financial)
| Tên cột (Feature) | Kiểu dữ liệu | Mô tả | Giá trị có thể có / Ví dụ |
| :--- | :--- | :--- | :--- |
| **Total_Asset_Value_Mil** | Float | Tổng tài sản hiện có tại công ty chứng khoán (Triệu VNĐ) | `150.5`, `2000.0` |
| **Cash_Balance_Mil** | Float | Số dư tiền mặt hiện tại có thể giao dịch (Triệu VNĐ) | `50.0`, `0.0` |
| **Cash_to_Asset_Ratio** | Float | Tỷ trọng tiền mặt trên tổng tài sản (Đo lường sức mua) | `0.0` đến `1.0` |
| **Num_Bank_Accounts** | Integer | Số lượng tài khoản ngân hàng đã liên kết | `1` đến `4` |
| **Credit_Score** | Integer | Điểm tín dụng nội bộ hoặc từ CIC | `400` đến `850` |

### 🔄 Nhóm Lịch sử giao dịch (Transaction Behavior)
| Tên cột (Feature) | Kiểu dữ liệu | Mô tả | Giá trị có thể có / Ví dụ |
| :--- | :--- | :--- | :--- |
| **Transaction_Count_6M** | Integer | Số lượng lệnh giao dịch đã khớp trong 6 tháng qua | `0`, `15`, `50` |
| **Transaction_Count_3M** | Integer | Số lượng lệnh giao dịch đã khớp trong 3 tháng qua | `0`, `5`, `20` |
| **Days_Since_Last_Trade** | Integer | Số ngày trôi qua kể từ lần giao dịch gần nhất | `0` đến `180` |

### 📱 Nhóm Hành vi sử dụng App (App Behavior)
| Tên cột (Feature) | Kiểu dữ liệu | Mô tả | Giá trị có thể có / Ví dụ |
| :--- | :--- | :--- | :--- |
| **App_Login_Freq_30D** | Integer | Số lần đăng nhập ứng dụng trong 30 ngày qua | `0`, `15` |
| **App_Login_Freq_7D** | Integer | Số lần đăng nhập ứng dụng trong 7 ngày qua | `0`, `5` |
| **Login_Momentum** | Float | Tốc độ gia tăng đăng nhập (7D so với trung bình tuần của 30D). `>1` là tăng. | `0.5`, `1.2`, `2.0` |
| **Watchlist_Count** | Integer | Số lượng mã cổ phiếu đang nằm trong danh mục theo dõi | `0` đến `19` |

### 🎯 Nhóm Hành vi đặc thù với mã VCK (Item-Specific)
| Tên cột (Feature) | Kiểu dữ liệu | Mô tả | Giá trị có thể có / Ví dụ |
| :--- | :--- | :--- | :--- |
| **Is_VCK_In_Watchlist** | Integer | Có đưa mã VCK vào Watchlist hay không? | `0` (Không), `1` (Có) |
| **VCK_Page_Views_30D** | Integer | Số lượt xem trang chi tiết mã VCK trong 30 ngày qua | `0`, `3`, `10` |
| **VCK_Page_Views_7D** | Integer | Số lượt xem trang chi tiết mã VCK trong 7 ngày qua | `0`, `2`, `5` |
| **Past_VCK_Transactions** | Integer | Số lần đã từng giao dịch mã VCK trong quá khứ | `0`, `1`, `5` |
| **Sector_Trade_Ratio** | Float | Tỷ trọng giao dịch các mã cùng ngành với VCK trong quá khứ | `0.0` đến `1.0` |

### ⚖️ Nhóm Khẩu vị rủi ro & Danh mục (Risk Profiling)
| Tên cột (Feature) | Kiểu dữ liệu | Mô tả | Giá trị có thể có / Ví dụ |
| :--- | :--- | :--- | :--- |
| **Current_Margin_Ratio** | Float | Tỷ lệ sử dụng margin hiện tại của tài khoản | `0.0` đến `1.0` |
| **Portfolio_Concentration** | Integer | Số lượng mã cổ phiếu đang nắm giữ (đo lường độ phân tán) | `1` đến `14` |
| **Avg_Holding_Period_Days** | Integer | Thời gian nắm giữ một mã cổ phiếu trung bình | `3` (Trader T+) đến `364` (Holder) |

---

## 3. Biến mục tiêu (Target Variables)

Dữ liệu cung cấp 2 phương án biến mục tiêu tùy thuộc vào yêu cầu bài toán kinh doanh. Chỉ sử dụng **MỘT** trong hai cột này trong quá trình huấn luyện mô hình (Xóa cột còn lại).

| Tên cột (Feature) | Kiểu dữ liệu | Mô tả | Giá trị có thể có / Ví dụ |
| :--- | :--- | :--- | :--- |
| **Target_Binary** | Integer | Bài toán phân loại nhị phân (Đã mua hay Chưa mua) | `1`: Mua (Top 20,000 khách hàng), `0`: Không mua |
| **Target_MultiClass**| Category | Bài toán phân loại đa lớp (Đánh giá mức độ tiềm năng) | `High`: Cao (Top 25,000)<br>`Medium`: Trung bình (60,000 tiếp theo)<br>`Low`: Thấp (Còn lại) |

---
*Lưu ý: Dữ liệu này được tạo hoàn toàn bằng các hàm ngẫu nhiên có kiểm soát (Mockup Data) phục vụ cho việc xây dựng Pipeline và thử nghiệm mô hình Machine Learning.*