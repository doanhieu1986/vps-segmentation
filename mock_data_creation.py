import pandas as pd
import numpy as np

# ---------------------------------------------------------
# 1. CẤU HÌNH BAN ĐẦU
# ---------------------------------------------------------
N_CUSTOMERS = 300000
N_BUYERS = 20000
np.random.seed(42) # Đảm bảo kết quả cố định mỗi lần chạy

print("Đang khởi tạo dữ liệu mockup...")

# ---------------------------------------------------------
# 2. TẠO CÁC NHÓM FEATURE CƠ BẢN (RAW DATA)
# ---------------------------------------------------------
df = pd.DataFrame({
    'Customer_ID': [f'CUST_{i:06d}' for i in range(1, N_CUSTOMERS + 1)],
    
    # --- Nhóm 1: Demographic ---
    'Age': np.random.randint(18, 75, N_CUSTOMERS),
    'Gender': np.random.choice(['M', 'F'], N_CUSTOMERS, p=[0.55, 0.45]),
    'City_Tier': np.random.choice([1, 2, 3], N_CUSTOMERS, p=[0.5, 0.3, 0.2]), # 1: HN/HCM, 2: Tỉnh lớn, 3: Khác
    'Income_Monthly_Mil': np.random.lognormal(mean=3.0, sigma=0.6, size=N_CUSTOMERS).round(1),
    'Occupation': np.random.choice(['Office', 'Business', 'Freelance', 'Student/Retired'], N_CUSTOMERS, p=[0.5, 0.3, 0.1, 0.1]),
    
    # --- Nhóm 2: Asset / Financial ---
    'Total_Asset_Value_Mil': np.random.lognormal(mean=5.0, sigma=1.2, size=N_CUSTOMERS).round(1),
    'Num_Bank_Accounts': np.random.randint(1, 5, N_CUSTOMERS),
    'Credit_Score': np.random.randint(400, 850, N_CUSTOMERS),
    
    # --- Nhóm 3: Transaction Behavior ---
    'Transaction_Count_6M': np.random.poisson(lam=20, size=N_CUSTOMERS),
    'Days_Since_Last_Trade': np.random.randint(0, 180, N_CUSTOMERS),
    
    # --- Nhóm 4: App Behavior ---
    'App_Login_Freq_30D': np.random.poisson(lam=15, size=N_CUSTOMERS),
    'Watchlist_Count': np.random.randint(0, 20, N_CUSTOMERS),
    
    # --- Nhóm 5: Item-Specific (Đặc thù mã VCK) ---
    'Is_VCK_In_Watchlist': np.random.choice([0, 1], N_CUSTOMERS, p=[0.85, 0.15]),
    'VCK_Page_Views_30D': np.random.poisson(lam=2, size=N_CUSTOMERS),
    'Past_VCK_Transactions': np.random.poisson(lam=0.5, size=N_CUSTOMERS),
    'Sector_Trade_Ratio': np.random.beta(a=2, b=5, size=N_CUSTOMERS).round(2), # Tỷ lệ giao dịch cùng ngành
    
    # --- Nhóm 6: Risk Profiling ---
    'Current_Margin_Ratio': np.random.beta(a=1, b=3, size=N_CUSTOMERS).round(2),
    'Portfolio_Concentration': np.random.randint(1, 15, N_CUSTOMERS), # Số mã đang cầm
    'Avg_Holding_Period_Days': np.random.randint(3, 365, N_CUSTOMERS)
})

# ---------------------------------------------------------
# 3. RÀNG BUỘC LOGIC & FEATURE ENGINEERING (TẠO RATIOS)
# ---------------------------------------------------------
# Ràng buộc Logic: Tài sản phải phù hợp với thu nhập (Cộng thêm base để tránh tài sản quá vô lý)
df['Total_Asset_Value_Mil'] = df['Total_Asset_Value_Mil'] + df['Income_Monthly_Mil'] * 12 * np.random.uniform(0.5, 2.0, N_CUSTOMERS)

# Tính Cash Balance: Tiền mặt luôn nhỏ hơn hoặc bằng Tổng tài sản
df['Cash_Balance_Mil'] = df['Total_Asset_Value_Mil'] * np.random.beta(a=2, b=5, size=N_CUSTOMERS)
df['Cash_Balance_Mil'] = df['Cash_Balance_Mil'].round(1)

# Tính tỷ lệ tiền mặt trên tổng tài sản (Cash_to_Asset_Ratio)
# Cộng 1e-5 để tránh lỗi chia cho 0 (Division by Zero)
df['Cash_to_Asset_Ratio'] = (df['Cash_Balance_Mil'] / (df['Total_Asset_Value_Mil'] + 1e-5)).round(3)

# Ràng buộc Logic App Login: 7 ngày phải <= 30 ngày
df['App_Login_Freq_7D'] = (df['App_Login_Freq_30D'] * np.random.uniform(0.1, 0.5, N_CUSTOMERS)).astype(int)

# Tính Momentum đăng nhập (Login_Momentum)
# Trung bình 1 tuần có (30D / 4) logins. Nếu 7D > trung bình -> Momentum > 1 (Đang active mạnh)
df['Login_Momentum'] = (df['App_Login_Freq_7D'] / ((df['App_Login_Freq_30D'] + 1e-5) / 4)).round(2)

# Ràng buộc Logic VCK Views: 7 ngày phải <= 30 ngày
df['VCK_Page_Views_7D'] = (df['VCK_Page_Views_30D'] * np.random.uniform(0.2, 0.8, N_CUSTOMERS)).astype(int)
# Ai có VCK trong Watchlist thì Views trung bình sẽ cao hơn
df.loc[df['Is_VCK_In_Watchlist'] == 1, 'VCK_Page_Views_30D'] += np.random.randint(3, 10, sum(df['Is_VCK_In_Watchlist'] == 1))

# Ràng buộc Logic Transaction: 3 tháng phải <= 6 tháng
df['Transaction_Count_3M'] = (df['Transaction_Count_6M'] * np.random.uniform(0.3, 0.6, N_CUSTOMERS)).astype(int)


# ---------------------------------------------------------
# 4. THIẾT LẬP TARGET (LABELING) DỰA TRÊN LATENT SCORE
# ---------------------------------------------------------
# Điểm tiềm năng mua VCK được xây dựng dựa trên trọng số của các features
# Trọng số cao nhất dành cho các hành vi liên quan trực tiếp đến VCK và Tỷ lệ tiền mặt
latent_score = (
    df['VCK_Page_Views_7D'] * 5.0 +               # Gần đây xem nhiều VCK (Tín hiệu rất mạnh)
    df['Is_VCK_In_Watchlist'] * 4.0 +             # Đã cho vào danh mục theo dõi
    df['Past_VCK_Transactions'] * 3.0 +           # Đã từng mua bán VCK (Quen thuộc)
    df['Sector_Trade_Ratio'] * 10.0 +             # Hay đánh dòng cổ phiếu này
    df['Cash_to_Asset_Ratio'] * 8.0 +             # Có nhiều tiền mặt chờ giải ngân
    df['Login_Momentum'] * 2.0 +                  # Đang mở app nhiều đột biến
    (180 - df['Days_Since_Last_Trade']) * 0.05 +  # Giao dịch càng gần đây điểm càng cao
    np.random.normal(0, 3, N_CUSTOMERS)           # Thêm nhiễu ngẫu nhiên (Noise)
)

df['Latent_Score'] = latent_score

# Sắp xếp để gán Label
df = df.sort_values(by='Latent_Score', ascending=False).reset_index(drop=True)

# Phương án 1: Binary (0/1)
df['Target_Binary'] = 0
df.loc[:N_BUYERS-1, 'Target_Binary'] = 1

# Phương án 2: Multi-class (High, Medium, Low)
df['Target_MultiClass'] = 'Low'
df.loc[:25000-1, 'Target_MultiClass'] = 'High'          # Top 25k cao nhất
df.loc[25000:85000-1, 'Target_MultiClass'] = 'Medium'   # 60k tiếp theo

# Xáo trộn ngẫu nhiên lại tập dữ liệu (Shuffle)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Xóa cột Latent Score vì không dùng trong model thực tế
df = df.drop(columns=['Latent_Score'])

# ---------------------------------------------------------
# 5. XUẤT THÔNG TIN KIỂM TRA
# ---------------------------------------------------------
print("\n--- PHÂN PHỐI TARGET BINARY ---")
print(df['Target_Binary'].value_counts(normalize=True).round(3) * 100)

print("\n--- PHÂN PHỐI TARGET MULTI-CLASS ---")
print(df['Target_MultiClass'].value_counts(normalize=True).round(3) * 100)

print("\n--- SAMPLE DATA DÀNH CHO KHÁCH HÀNG ĐÃ MUA (TARGET = 1) ---")
display_cols = ['Customer_ID', 'Cash_to_Asset_Ratio', 'Is_VCK_In_Watchlist', 'VCK_Page_Views_7D', 'Login_Momentum', 'Target_Binary']
print(df[df['Target_Binary'] == 1][display_cols].head())

# Lưu file (Bỏ comment dòng dưới để xuất ra file CSV)
df.to_csv('vck_propensity_mockup.csv', index=False)
print("\nHoàn tất tạo bộ dữ liệu 300,000 dòng!")