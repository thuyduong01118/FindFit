import streamlit as st
import pandas as pd
from deep_translator import GoogleTranslator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image

from sentence_transformers import SentenceTransformer, util
# import gensim 
# from gensim.models import Word2Vec




# Cài đặt giao diện trang
st.set_page_config(layout="wide")

# Load ảnh
logo1 = Image.open("E:\\SIC\\app\\logo_dhm.png")
logo2 = Image.open("E:\\SIC\\app\\CNTT.png")
logo3 = Image.open("E:\\SIC\\app\\SIC.jpg")

# Hiển thị logo theo hàng ngang
col1, col2, col3 = st.columns(3)

with col1:
    
    st.markdown(
        "<div style='text-align: center; font-size:22px; margin-top:10px;'>"
        "Trường Đại học Mở Thành phố Hồ Chí Minh"
        "</div>", unsafe_allow_html=True)
    st.image(logo1, use_container_width=True)
with col2:
    
    st.markdown(
        "<div style='text-align: center; font-size:22px; margin-top:10px;'>"
        "Khoa Công nghệ Thông tin"
        "</div>", unsafe_allow_html=True)
    st.image(logo2, use_container_width=True)
with col3:
    
    st.markdown(
        "<div style='text-align: center; font-size:22px; margin-top:10px;'>"
        "Samsung Innovation Campus"
        "</div>", unsafe_allow_html=True)
    st.image(logo3, use_container_width=True)
# Thêm tiêu đề chính
st.markdown("<h2 style='text-align: center; color: #4a90e2;'>Hệ thống Gợi ý Việc làm</h2>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: gray;'>Thực hiện bởi Nhóm 3 </h4>", unsafe_allow_html=True)



# Load dữ liệu
@st.cache_data
def load_data():
    return pd.read_excel("E:\\SIC\\app\\jobs_clean.xlsx")

df = load_data()
#---------------------
import joblib

rf_model = joblib.load("E:/SIC/app/career_rf_model.pkl")
vectorizer = joblib.load("E:/SIC/app/tfidf_vectorizer.pkl")
label_encoder = joblib.load("E:/SIC/app/label_encoder.pkl")

#--------

# ========== 1. Chọn ngành nghề ==========
# industries = df["nganh_nghe_clean"].dropna().unique().tolist()
# industry_choice = st.selectbox("📌 Chọn ngành nghề:", sorted(industries))

# df_industry = df[df["nganh_nghe_clean"] == industry_choice].copy()
df_industry = df.copy()


# Lấy toàn bộ dữ liệu từ cột khu_vuc
all_locations = df["khu_vuc"].dropna().astype(str).tolist()

# Tách theo dấu phẩy rồi gom lại thành 1 list duy nhất
split_locations = []
for loc in all_locations:
    split_locations.extend([x.strip() for x in loc.split(",")])

# Lấy unique + sort
tat_ca = sorted(set(split_locations))

# Checkbox chọn tất cả
chon_tat_ca = st.checkbox("✅ Chọn tất cả khu vực", value=False)

if chon_tat_ca:
    khu_vuc_chon = st.multiselect("📍 Chọn khu vực làm việc:", options=tat_ca, default=tat_ca)
else:
    khu_vuc_chon = st.multiselect("📍 Chọn khu vực làm việc:", options=tat_ca)

#st.write("✅ Khu vực đã chọn:", khu_vuc_chon)

# Lọc theo khu vực đã chọn
pattern = '|'.join(khu_vuc_chon)
df_location = df_industry[df_industry["khu_vuc"].fillna("").str.contains(pattern, case=False)].copy()

# ========== 3. Chọn cấp bậc ==========
cap_bac_list = df_location["cap_bac_standardized"].dropna().unique().tolist()
cap_bac_choices = st.multiselect("🏷️ Chọn cấp bậc:", sorted(cap_bac_list), default=cap_bac_list)

# Lọc dữ liệu cuối cùng theo khu vực + cấp bậc
df_final = df_location[df_location["cap_bac_standardized"].isin(cap_bac_choices)].copy()

# ========== 4. Nhập kinh nghiệm & kỹ năng ==========
experience = st.number_input("🎓 Số năm kinh nghiệm:", min_value=0, max_value=30, value=1)
skills_vi = st.text_input("🛠 Nhập kỹ năng (cách nhau bởi dấu phẩy):")

# ========== 5. Dịch kỹ năng ==========
def translate_vi_to_en(text):
    if text.strip() == "":
        return ""
    try:
        return GoogleTranslator(source='vi', target='en').translate(text)
    except Exception as e:
        st.error(f"Lỗi dịch: {e}")
        return text
if st.button("🤖 Gợi ý ngành và công việc phù hợp"):
    if skills_vi.strip() == "":
        st.warning("❗ Bạn cần nhập kỹ năng.")
    elif not khu_vuc_chon:
        st.warning("❗ Bạn cần chọn ít nhất một khu vực.")
    elif not cap_bac_choices:
        st.warning("❗ Bạn cần chọn cấp bậc.")
    else:
        with st.spinner("🚀 Đang xử lý..."):
            # ======= 1. Dịch kỹ năng và dự đoán ngành =======
            skills_en = translate_vi_to_en(skills_vi)
            skill_vec = vectorizer.transform([skills_en])
            pred = rf_model.predict(skill_vec)
            predicted_career = label_encoder.inverse_transform(pred)[0]

            st.success(f"🌟 Ngành nghề phù hợp: **{predicted_career}**")

            # ======= 2. Lọc công việc theo ngành, khu vực, cấp bậc =======
            df_filtered = df.copy()
            pattern = '|'.join(khu_vuc_chon)
            df_filtered = df_filtered[df_filtered["khu_vuc"].fillna("").str.contains(pattern, case=False)]
            df_filtered = df_filtered[df_filtered["cap_bac_standardized"].isin(cap_bac_choices)]
            df_filtered = df_filtered[df_filtered["nganh_nghe_clean"] == predicted_career]

            if df_filtered.empty:
                st.warning("😥 Không tìm thấy công việc phù hợp với ngành và tiêu chí đã chọn.")
            else:
                # ======= 3. Tính similarity bằng Sentence-BERT =======
                model = SentenceTransformer('all-MiniLM-L6-v2')
                job_skills = df_filtered["ky_nang_combined"].fillna("").tolist()
                job_embeddings = model.encode(job_skills, convert_to_tensor=True)
                user_embedding = model.encode(skills_en, convert_to_tensor=True)

                sims = util.cos_sim(user_embedding, job_embeddings)[0].cpu().numpy()
                df_filtered["similarity_skills"] = sims

                # ======= 4. Chọn Top 10 công việc phù hợp =======
                top_jobs = df_filtered.sort_values(by="similarity_skills", ascending=False).head(10)

                # ======= 5. Hiển thị kết quả =======
                st.subheader("✅ Top 10 công việc phù hợp với bạn:")
                # Tạo cột phần trăm từ similarity_skills
                top_jobs["muc_do_phu_hop(%)"] = (top_jobs["similarity_skills"] * 100).round(1).astype(str) + "%"
                # Lấy các cột hiển thị
                top_jobs_display = top_jobs[[ "ten_cong_viec", "khu_vuc", "cap_bac_standardized", "muc_do_phu_hop(%)", "nganh_nghe_clean", "link"]].rename(columns={
                    "ten_cong_viec": "Tên công việc","khu_vuc": "Khu vực","cap_bac_standardized": "Cấp bậc","muc_do_phu_hop(%)": "Mức độ phù hợp","nganh_nghe_clean": "Ngành nghề",
                    "link": "Link tuyển dụng"})
                # Thêm cột STT thủ công
                top_jobs_display.reset_index(drop=True, inplace=True)
                top_jobs_display.insert(0, "STT", range(1, len(top_jobs_display) + 1))
                # Hiển thị mà không hiển thị index mặc định
                st.dataframe(top_jobs_display, use_container_width=True, hide_index=True)

# #-------------------------
# if st.button("📊 Dự đoán ngành nghề phù hợp (ML)"):
#     if skills_vi.strip() == "":
#         st.warning("Bạn cần nhập kỹ năng.")
#     else:
#         skills_en = translate_vi_to_en(skills_vi)
#         skill_vec = vectorizer.transform([skills_en])
#         pred = rf_model.predict(skill_vec)
#         predicted_career = label_encoder.inverse_transform(pred)[0]

#         st.success(f"🌟 Ngành nghề được gợi ý: **{predicted_career}**")
# #-------------------


# # ========== 6. Tính TF-IDF + Cosine Similarity ==========
# if st.button("🔎 Tìm việc theo kỹ năng"):
#     if df_final.empty:
#         st.warning("Không tìm thấy công việc phù hợp với khu vực và cấp bậc đã chọn.")
#     else:
#         skills_en = translate_vi_to_en(skills_vi)
#         st.success(f"👉 Kỹ năng đã dịch sang EN: **{skills_en}**")

#         # # TF-IDF cho kỹ năng
#         # job_skills = df_final["ky_nang_combined"].fillna("").tolist()
#         # tfidf = TfidfVectorizer(max_features=5000)
#         # tfidf_matrix = tfidf.fit_transform(job_skills)
#         # user_vec = tfidf.transform([skills_en])
#         # sims = cosine_similarity(user_vec, tfidf_matrix).flatten()

#         # df_final["similarity_skills"] = sims
#         # top_jobs = df_final.sort_values(by="similarity_skills", ascending=False).head(10)

#         # st.subheader("✅ Top 10 công việc phù hợp với kỹ năng")
#         # st.dataframe(top_jobs[[
#         #     "ten_cong_viec", "khu_vuc", "cap_bac_standardized", 
#         #     "similarity_skills", "nganh_nghe_clean","link"
#         # ]])

#         #Load mô hình Sentence-BERT
#         with st.spinner("🔍 Đang tính toán độ tương đồng..."):
#             model = SentenceTransformer('all-MiniLM-L6-v2')

#             job_skills = df_final["ky_nang_combined"].fillna("").tolist()
#             job_embeddings = model.encode(job_skills, convert_to_tensor=True)
#             user_embedding = model.encode(skills_en, convert_to_tensor=True)

#             sims = util.cos_sim(user_embedding, job_embeddings)[0].cpu().numpy()
#             df_final["similarity_skills"] = sims
#             top_jobs = df_final.sort_values(by="similarity_skills", ascending=False).head(10)

#         st.subheader("✅ Top 10 công việc phù hợp với kỹ năng")
#         st.dataframe(top_jobs[[ 
#             "ten_cong_viec", "khu_vuc", "cap_bac_standardized", 
#             "similarity_skills", "nganh_nghe_clean", "link"
#         ]])

#         # #w2_vec
#         # # Vector hóa kỹ năng người dùng bằng Word2Vec
#         # user_vec = sentence_vector(skills_en, word_vectors)

#         # job_skills = df_final["ky_nang_combined"].fillna("").tolist()
#         # job_vecs = [sentence_vector(s, word_vectors) for s in job_skills]

#         # # Tính cosine similarity
#         # def cosine_sim(v1, v2):
#         #     if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
#         #         return 0
#         #     return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

#         # sims = [cosine_sim(user_vec, job_vec) for job_vec in job_vecs]
#         # df_final["similarity_skills"] = sims

#         # top_jobs = df_final.sort_values(by="similarity_skills", ascending=False).head(10)
#         # st.subheader("✅ Top 10 công việc phù hợp với kỹ năng (Word2Vec)")
#         # st.dataframe(top_jobs[[
#         #     "ten_cong_viec", "khu_vuc", "cap_bac_standardized", 
#         #     "similarity_skills", "nganh_nghe_clean","link"
#         # ]])
    

