import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# 1. Load dữ liệu
df = pd.read_excel("E:/SIC/app/jobs_clean.xlsx")

# 2. Chuẩn bị dữ liệu
X_text = df["ky_nang_combined"].fillna("")          # text kỹ năng
y_labels = df["nganh_nghe_clean"].fillna("Khác")    # nhãn ngành nghề

# 3. Encode nhãn
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_labels)

# 4. Vector hóa TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(X_text)

# 5. Chia train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Train Random Forest
rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
rf_model.fit(X_train, y_train)

# 7. Lưu model + vectorizer + label encoder
joblib.dump(rf_model, "E:/SIC/app/career_rf_model.pkl")
joblib.dump(vectorizer, "E:/SIC/app/tfidf_vectorizer.pkl")
joblib.dump(label_encoder, "E:/SIC/app/label_encoder.pkl")

print("✅ Đã train và lưu mô hình thành công!")
