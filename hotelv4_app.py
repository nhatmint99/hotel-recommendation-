# app.py
# Streamlit: Recommender cho Agoda – Content-based (TF-IDF) + Collaborative Filtering (item–item)

import streamlit as st
import pandas as pd
import numpy as np
import re, io, os
from typing import List, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
import seaborn as sns
st.set_page_config(page_title="Hotel Recommender (Agoda)", layout="wide")

# ==========================
# Helpers
# ==========================
def clean_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace(",", ".", regex=False), errors="coerce")

def load_vi_stopwords(file: Optional[io.BytesIO]) -> set:
    sw = set()
    if file is None:
        # fallback: bundled or empty
        return sw
    txt = file.read().decode("utf-8", errors="ignore")
    for line in txt.splitlines():
        line = str(line).strip().lower()
        if not line or line.startswith("#"):
            continue
        for tok in line.replace("_"," ").split():
            sw.add(tok)
    return sw

EN_SW = set("""
the and for with you your from that this are was were have has had but not out our very into over under
then than too can could would should will just they them their there here when where what which who why
how about more most much many any some also been being we he she it its as on in to of at by a an or
if is be do did done me my i
""".split())
# Load custom stopwords
STOPWORDS_FILE = "combined_stopwords.txt"

def load_stopwords(path=STOPWORDS_FILE):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            stopwords = set()
            for line in f:
                # handle lines like "nyc    người yêu cũ"
                parts = line.strip().split()
                if parts:
                    stopwords.add(parts[0].strip())
        return stopwords
    return set()

CUSTOM_STOPWORDS = load_stopwords()
def clean_text(text: str, stopwords: set = CUSTOM_STOPWORDS) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"[^a-zA-ZÀ-Ỹà-ỹ0-9 ]", " ", text)  # keep accented Vietnamese chars
    text = text.lower()
    tokens = text.split()
    tokens = [t for t in tokens if t not in stopwords and len(t) > 1]
    return " ".join(tokens)
def tokenize(text, CUSTOM_STOPWORDS: set, en_sw: set = EN_SW) -> List[str]:
    if isinstance(text, list):
        text = " ".join(map(str, text))
    elif not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r"[^a-zA-ZÀ-ỹ\s]", " ", text)
    toks = [t for t in text.split() if t and (t not in CUSTOM_STOPWORDS) and (t not in en_sw)]
    return [t for t in toks if len(t) >= 2]

# ==========================
# Sidebar – Data & Settings
# ==========================
st.sidebar.header("⚙️ Cấu hình")
uploaded_comments = st.sidebar.file_uploader("Upload hotel_comments.csv", type=["csv"])
uploaded_info     = st.sidebar.file_uploader("Upload hotel_info.csv", type=["csv"])
uploaded_vi_sw    = st.sidebar.file_uploader("Upload vietnamese-stopwords.txt", type=["txt"])

@st.cache_data(show_spinner=False)
def load_datasets(u_comments, u_info) -> Tuple[pd.DataFrame, pd.DataFrame]:
    comments = pd.read_csv(u_comments) if u_comments else pd.read_csv("hotel_comments.csv")
    info = pd.read_csv(u_info) if u_info else pd.read_csv("hotel_info.csv")

    # normalize hotel ID
    if "Hotel ID" in comments.columns:
        comments.rename(columns={"Hotel ID": "Hotel_ID"}, inplace=True)

    comments["Score_clean"] = clean_numeric(comments.get("Score", pd.Series(index=comments.index))).clip(lower=0, upper=10)
    if "Review Date" in comments.columns:
        comments["Review Date"] = pd.to_datetime(comments["Review Date"].astype(str)
                                                 .str.replace("Đã nhận xét vào ", "", regex=False),
                                                 errors="coerce")

    for col in ["Total_Score","Location","Cleanliness","Service","Facilities","Value_for_money","Comfort_and_room_quality"]:
        if col in info.columns:
            info[col + "_clean"] = clean_numeric(info[col]).clip(lower=0, upper=10)

    comments["Hotel_ID"] = comments["Hotel_ID"].astype(str)
    if "Reviewer ID" in comments.columns and "Reviewer Name" in comments.columns:
        comments["Reviewer Label"] = (
            comments["Reviewer ID"].astype(str).str.split("_").str[-1] + " " + comments["Reviewer Name"].astype(str)
        )

    if "Hotel_ID" in info.columns:
        info["Hotel_ID"] = info["Hotel_ID"].astype(str)

    return comments, info

comments, info = load_datasets(uploaded_comments, uploaded_info)
VI_SW = load_vi_stopwords(uploaded_vi_sw)

st.sidebar.success(f"Loaded: comments={comments.shape}, info={info.shape}")

# ==========================
# Build per-hotel documents
# ==========================
@st.cache_data(show_spinner=False)
def build_hotel_docs(comments: pd.DataFrame, info: pd.DataFrame, vi_sw: set):
    info_text = info[["Hotel_ID","Hotel_Name","Hotel_Address","Hotel_Description"]].copy()
    info_text["Hotel_Description"] = info_text["Hotel_Description"].fillna("")

    reviews_text = comments.groupby("Hotel_ID").agg(
        titles=("Title", lambda x: " ".join(map(str, x.dropna().tolist()))[:200000]) if "Title" in comments else ("Title", " ".join),
        bodies=("Body",  lambda x: " ".join(map(str, x.dropna().tolist()))[:400000]) if "Body" in comments else ("Body", " ".join)
    ).reset_index() if "Title" in comments and "Body" in comments else comments.groupby("Hotel_ID").size().reset_index().rename(columns={0:"cnt"})

    hotels_text = info_text.merge(reviews_text, on="Hotel_ID", how="left")
    for col in ["titles","bodies"]:
        if col not in hotels_text:
            hotels_text[col] = ""
    hotels_text["doc_raw"] = (
        hotels_text["Hotel_Description"].astype(str) + " " +
        hotels_text["titles"].astype(str) + " " +
        hotels_text["bodies"].astype(str)
    )
    tokens_list = [tokenize(t, vi_sw) for t in hotels_text["doc_raw"].tolist()]
    hotels_text["doc"] = [" ".join(t) for t in tokens_list]
    return hotels_text, tokens_list

hotels_text, tokens_list = build_hotel_docs(comments, info, VI_SW)

# ==========================
# Content-based TF-IDF
# ==========================
@st.cache_resource(show_spinner=False)
def build_cb_index(docs):
    vectorizer = TfidfVectorizer(max_features=10000)
    tfidf_matrix = vectorizer.fit_transform(docs)
    return vectorizer, tfidf_matrix

vectorizer, tfidf_matrix = build_cb_index(hotels_text["doc"].tolist())

def cb_scores_from_text(query: str, top_k=20) -> pd.DataFrame:
    q_toks = " ".join(tokenize(query, VI_SW))
    q_vec = vectorizer.transform([q_toks])
    sims = cosine_similarity(q_vec, tfidf_matrix).flatten()
    order = np.argsort(-sims)[:top_k]
    rows = []
    for j in order:
        rows.append({
            "Hotel_ID": hotels_text.iloc[j].get("Hotel_ID", ""),
            "Hotel_Name": hotels_text.iloc[j].get("Hotel_Name", ""),
            "Hotel_Address": hotels_text.iloc[j].get("Hotel_Address", ""),
            "cb_score": float(sims[j])
        })
    return pd.DataFrame(rows)

def recommend_hotels(hotel_id: str, top_k: int = 5) -> pd.DataFrame:
    if hotel_id not in hotels_text["Hotel_ID"].values:
        return pd.DataFrame({"note": [f"{hotel_id} không có trong dữ liệu"]})

    idx = hotels_text.index[hotels_text["Hotel_ID"] == hotel_id][0]
    sims = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    order = np.argsort(-sims)[1:top_k+1]  # skip itself

    rows = []
    for j in order:
        rows.append({
            "Hotel_ID": hotels_text.iloc[j]["Hotel_ID"],
            "Hotel_Name": hotels_text.iloc[j]["Hotel_Name"],
            "Hotel_Address": hotels_text.iloc[j]["Hotel_Address"],
            "similarity": float(sims[j])
        })
    return pd.DataFrame(rows)

# ==========================
# Collaborative Filtering (item–item)
# ==========================
@st.cache_resource(show_spinner=False)
def build_item_item(comments: pd.DataFrame):
    if not {"Reviewer Label","Hotel_ID","Score_clean"}.issubset(comments.columns):
        return None
    ui = comments.pivot_table(index="Reviewer Label", columns="Hotel_ID", values="Score_clean", aggfunc="mean")
    if ui.empty:
        return None
    user_mean = ui.mean(axis=1)
    ui_centered = ui.sub(user_mean, axis=0)
    X = ui_centered.fillna(0).to_numpy(dtype=float)
    norms = np.linalg.norm(X, axis=0, keepdims=True)
    norms[norms==0] = 1.0
    Xn = X / norms
    item_sim = Xn.T @ Xn
    return {
        "ui": ui,
        "ui_centered": ui_centered,
        "item_sim": item_sim,
        "item_ids": ui.columns.tolist(),
        "user_mean": user_mean
    }

cf_state = build_item_item(comments)

def cf_recommend_for_user(reviewer_label: str, top_k=20) -> pd.DataFrame:
    if cf_state is None or reviewer_label not in cf_state["ui"].index:
        return pd.DataFrame()
    ui, ui_centered, item_sim, item_ids, user_mean = (
        cf_state["ui"], cf_state["ui_centered"], cf_state["item_sim"], cf_state["item_ids"], cf_state["user_mean"]
    )
    seen_mask = ui.loc[reviewer_label].notna().values
    preds = item_sim @ ui_centered.loc[reviewer_label].fillna(0).to_numpy(dtype=float)
    preds[seen_mask] = -1e9
    top_idx = np.argsort(-preds)[:top_k]
    id_name = info[["Hotel_ID","Hotel_Name","Hotel_Address"]].drop_duplicates()
    df = pd.DataFrame({"Hotel_ID":[item_ids[j] for j in top_idx],
                       "cf_score": [float(preds[j]) for j in top_idx]})
    df = df.merge(id_name, on="Hotel_ID", how="left")
    return df

# ==========================
# MENU
# ==========================
menu = st.sidebar.radio("📑 Menu", [
    "Introduction",
    "Business Problem",
    "Evaluation & Report",
    "Insight",
    "Recommendation / New Prediction"
])

# ==========================
# 1) Introduction
# ==========================
if menu == "Introduction":
    st.title("Hotel Recommendation – Agoda (Nhóm H - Trần Nhật Minh)")
    st.markdown("""
**Mục tiêu**: Gợi ý khách sạn theo **nội dung mô tả + đánh giá** (TF-IDF + cosine), kết hợp **CF item-item** khi có chồng chéo người dùng, và cung cấp EDA/tương tác dự đoán cho dữ liệu mới.

**Dữ liệu**:
- `hotel_comments.csv`: đánh giá/điểm/tiêu đề/nội dung theo người dùng – khách sạn.
- `hotel_info.csv`: thông tin khách sạn, mô tả, địa chỉ, các tiêu chí (Location, Cleanliness, ...).
- `vietnamese-stopwords.txt`: stopwords tiếng Việt để làm sạch văn bản.

Bạn có thể upload thay thế file ở **Sidebar**.
    """)
    st.image("Agoda_img.jpg",
         caption="Agoda – Online Travel Booking", use_column_width=True)

    # ==========================
    # 2) Business Problem
    # ==========================
elif menu == "Business Problem":
    st.header("Bài toán kinh doanh")
    st.markdown("""
**Agoda** là một trang web đặt phòng trực tuyến có trụ sở tại Singapore, được thành lập vào năm 2005, thuộc sở hữu của **Booking Holdings Inc.**  
- Agoda chuyên cung cấp dịch vụ đặt phòng khách sạn, căn hộ, nhà nghỉ và các loại hình lưu trú trên toàn cầu.  
- Trang web cho phép người dùng **tìm kiếm, so sánh và đặt chỗ** ở với **mức giá ưu đãi**.

**Yêu cầu đề xuất**: cung cấp khuyến nghị khách sạn **phù hợp sở thích** của người dùng hoặc **phù hợp bộ lọc** (sao, địa chỉ, từ khóa), tối ưu **chuyển đổi** trong bối cảnh dữ liệu đánh giá **thưa**.
    """)
    st.image("hotel_review.img", caption="Hotel Reviews", use_column_width=True)

elif menu == "Evaluation & Report":
    st.header("📊 Evaluation & Report")
    st.metric("Số review", f"{len(comments):,}")
    st.metric("Số khách sạn", f"{info['Hotel_ID'].nunique():,}")

    col1, col2 = st.columns(2)
    with col1:
        if "Reviewer_Label" in comments:
            st.metric("Số người dùng", f"{comments['Reviewer_Label'].nunique():,}")
    with col2:
        st.write("Phân phối điểm review (Score_clean)")
        sc = comments["Score_clean"].dropna()
        st.bar_chart(pd.DataFrame({"count": sc.value_counts().sort_index()}))

    # Top hotels by review count
    st.subheader("🏨 Top khách sạn theo số lượng review")

    # Count reviews per hotel
    hotel_review_counts = comments["Hotel_ID"].value_counts().head(20).reset_index()
    hotel_review_counts.columns = ["Hotel_ID", "count"]

    # Merge to get names
    hotel_review_counts = hotel_review_counts.merge(
        info[["Hotel_ID", "Hotel_Name"]],
        on="Hotel_ID", how="left"
    )

    # Use Hotel_Name as index
    hotel_review_counts = hotel_review_counts.set_index("Hotel_Name").sort_values("count", ascending=True)

    # Plot
    st.bar_chart(hotel_review_counts["count"])


    # Boxplot theo quốc tịch
    if "Nationality" in comments.columns:
        st.subheader("📦 Boxplot điểm theo quốc tịch (top 10)")
        fig, ax = plt.subplots(figsize=(10,5))
        top_nat = comments["Nationality"].value_counts().head(10).index
        subset = comments[comments["Nationality"].isin(top_nat)]
        sns.boxplot(x="Nationality", y="Score_clean", data=subset, ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        st.pyplot(fig)

    # Correlation heatmap
    st.subheader("📈 Tương quan tiêu chí với Total_Score")
    sub_cols = [c for c in ["Location_clean","Cleanliness_clean","Service_clean",
                            "Facilities_clean","Value_for_money_clean","Comfort_and_room_quality_clean"]
                if c in info.columns]
    if sub_cols and "Total_Score_clean" in info.columns:
        corr = info[["Total_Score_clean"] + sub_cols].corr()
        fig, ax = plt.subplots(figsize=(8,6))

        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    st.divider()

    # ---- Modeling summary ----
    st.subheader("Modeling")
    st.markdown("""
- **Content-based**: TF-IDF + cosine từ **mô tả + tiêu đề + nội dung review** theo khách sạn.  
- **Collaborative (item–item)**: dựa trên ma trận user-item mean-centered; dùng khi có chồng chéo).
    """)
    st.markdown("**Lưu ý**: do dữ liệu đánh giá thưa, nên ưu tiên CB (mô tả + review) để gợi ý khách sạn mới hoặc không có chồng chéo người dùng.")
    st.markdown("""
**Đánh giá**: do không có tập test, nên đánh giá sơ bộ bằng cách xem xét thủ công kết quả gợi ý (qualitative).  
- Với truy vấn văn bản, kết quả gợi ý khá hợp lý (xem tab New Prediction / Analysis).  
- Với người dùng có chồng chéo, kết quả gợi ý cũng khá hợp lý (xem tab New Prediction / Analysis).  
- Với người dùng không có chồng chéo, kết quả gợi ý chỉ dựa trên CB.
    """)
    st.markdown("**Cải tiến**: có thể kết hợp thêm các yếu tố khác như rank, địa chỉ, giá, ... để lọc/gợi ý phù hợp hơn.")
    

elif menu == "Insight":
    st.header("📊 Insight cho từng khách sạn")
    # hotel_name_to_id = dict(zip(info["Hotel_Name"], info["Hotel_ID"]))
    # selected_hotel_name = st.selectbox("Chọn khách sạn", sorted(info["Hotel_Name"].dropna().unique().tolist()))
    # selected_hotel_id = hotel_name_to_id[selected_hotel_name]
    # st.markdown(f"### 🏨 {selected_hotel_name}")
    # st.caption(f"Hotel_ID: {selected_hotel_id}")
    tabs = st.tabs(["Select by Hotel Name", "By Keyword Search"])
    with tabs[0]:
        hotel_name_to_id = dict(zip(info["Hotel_Name"], info["Hotel_ID"]))
        selected_hotel_name = st.selectbox("Chọn khách sạn", sorted(info["Hotel_Name"].dropna().unique().tolist()))
        selected_hotel_id = hotel_name_to_id[selected_hotel_name]
        st.markdown(f"### 🏨 {selected_hotel_name}")
        st.caption(f"Hotel_ID: {selected_hotel_id}")
        hotel_reviews = comments[comments["Hotel_ID"] == selected_hotel_id].copy()
        hotel_info = info[info["Hotel_ID"] == selected_hotel_id].copy()

        if hotel_reviews.empty:
            st.warning("❌ Không có review cho khách sạn này.")
            st.stop()

        # --- Overview metrics ---
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Số review", f"{len(hotel_reviews):,}")
        with col2:
            st.metric("Điểm trung bình", f"{hotel_reviews['Score_clean'].mean():.2f}")
        with col3:
            st.metric("Điểm mới nhất", f"{hotel_reviews['Score_clean'].iloc[-1]:.1f}")

        # --- Address / Location ---
        st.subheader("📍 Địa chỉ khách sạn")
        if "Hotel_Address" in hotel_info.columns:
            st.write(hotel_info["Hotel_Address"].iloc[0])
        else:
            st.info("Không có cột Hotel_Address trong hotel_info.csv")


        # --- Trend over time ---
        # if "Review Date" in hotel_reviews.columns:
        #     hotel_reviews["year_month"] = hotel_reviews["Review Date"].dt.to_period("M").astype(str)
        #     trend = hotel_reviews.groupby("year_month")["Score_clean"].agg(["count","mean"]).reset_index()

        #     st.subheader("📈 Xu hướng review & điểm trung bình theo tháng")
        #     import matplotlib.pyplot as plt
        #     fig, ax1 = plt.subplots(figsize=(12,5))
        #     ax2 = ax1.twinx()
        #     ax1.bar(trend["year_month"], trend["count"], color="skyblue", alpha=0.6)
        #     ax2.plot(trend["year_month"], trend["mean"], color="red", marker="o")
        #     ax1.set_ylabel("Số review")
        #     ax2.set_ylabel("Điểm trung bình")
        #     ax1.set_xticks(range(0,len(trend), max(1,len(trend)//12)))
        #     ax1.set_xticklabels(trend["year_month"][::max(1,len(trend)//12)], rotation=45, ha="right")
        #     st.pyplot(fig)

        # --- Nationality mix ---
        if "Nationality" in hotel_reviews.columns:
            st.subheader("🌍 Quốc tịch reviewer")
            nat_counts = hotel_reviews["Nationality"].value_counts().head(10)
            fig, ax = plt.subplots(figsize=(8,5))
            ax.barh(nat_counts.index[::-1], nat_counts.values[::-1], color="green")
            ax.set_xlabel("Số review")
            st.pyplot(fig)

        # --- Group mix ---
        if "Group" in hotel_reviews.columns or "Traveler type" in hotel_reviews.columns:
            st.subheader("👥 Nhóm khách / loại chuyến đi")
            group_col = "Group" if "Group" in hotel_reviews.columns else "Traveler type"
            grp_counts = hotel_reviews[group_col].value_counts().head(6)
            fig, ax = plt.subplots()
            ax.pie(grp_counts.values, labels=grp_counts.index, autopct='%1.1f%%', startangle=90)
            ax.axis("equal")
            st.pyplot(fig)

        # --- Distribution of scores ---
        st.subheader("📊 Phân phối điểm review")
        fig, ax = plt.subplots()
        ax.hist(hotel_reviews["Score_clean"].dropna(), bins=10, color="skyblue", edgecolor="black")
        ax.set_xlabel("Score")
        ax.set_ylabel("Count")
        st.pyplot(fig)

        # --- WordClouds ---
        st.subheader("☁️ WordCloud từ review")
        from wordcloud import WordCloud
        def make_wc(texts, title):
            text = " ".join([" ".join(tokenize(t, VI_SW)) for t in texts])
            if not text.strip():
                st.info(f"Không có dữ liệu cho {title}")
                return
            wc = WordCloud(width=400, height=200, background_color="white").generate(text)
            fig, ax = plt.subplots(figsize=(10,5))
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig)
        pos_texts = (hotel_reviews["Title"].fillna("") + " " + hotel_reviews["Body"].fillna("")).loc[hotel_reviews["Score_clean"]>=8].tolist()
        neg_texts = (hotel_reviews["Title"].fillna("") + " " + hotel_reviews["Body"].fillna("")).loc[hotel_reviews["Score_clean"]<=7.9].tolist()
        make_wc(pos_texts, "Review tích cực (≥9)")
        make_wc(neg_texts, "Review tiêu cực (≤7)")

        if hotel_reviews.empty:
            st.warning("Không có review cho khách sạn này.")
        else:
            st.metric("Số review", f"{len(hotel_reviews):,}")
            st.metric("Điểm TB", f"{hotel_reviews['Score_clean'].mean():.2f}")
    with tabs[1]:
        st.info("Tìm kiếm khách sạn theo từ khóa trong mô tả hoặc review")
        query = st.text_input("Nhập từ khóa", value="beach resort")
        if query.strip():
            results = cb_scores_from_text(query, top_k=20)
            if results.empty:
                st.warning("Không tìm thấy khách sạn phù hợp.")
            else:
                st.dataframe(results)
    # Define hotels_filtered based on available hotels_text or info DataFrame
        hotels_filtered = hotels_text if not hotels_text.empty else info
        if not hotels_filtered.empty:
            hotel_id_to_name = dict(zip(hotels_filtered["Hotel_ID"], hotels_filtered.get("Hotel_Name", hotels_filtered["Hotel_ID"])))
            selected_hotel_id = st.selectbox("Chọn Hotel_ID để phân tích", sorted(hotels_filtered["Hotel_ID"].dropna().unique().tolist()))
            hotel_id = selected_hotel_id
            st.markdown(f"### 🏨 {hotel_id_to_name.get(hotel_id, hotel_id)}")
            h_ratings = comments[comments["Hotel_ID"].astype(str) == hotel_id]

            # --- Hotel Info ---
            st.subheader("🏨 Hotel Information")
            hrow = info[info["Hotel_ID"].astype(str) == hotel_id].head(1)
            if not hrow.empty:
                info_cols = [c for c in ["Hotel_Name", "City", "Address", "Amenities", "Description"] if c in hrow.columns]
                st.write({c: hrow.iloc[0][c] for c in info_cols})

            # --- Ratings Analysis ---
            st.subheader("⭐ Ratings Analysis")
            colsM = st.columns(2)
            with colsM[0]:
                st.metric("Average Score", f"{h_ratings['Score_clean'].mean():.2f}" if not h_ratings.empty else "N/A")
            with colsM[1]:
                st.metric("Number of Ratings", len(h_ratings))

            if not h_ratings.empty:
                fig_r, ax_r = plt.subplots()
                h_ratings["Score_clean"].hist(bins=10, edgecolor="black", ax=ax_r)
                ax_r.set_xlabel("Score_clean"); ax_r.set_ylabel("Count")
                ax_r.set_title("Rating Distribution")
                st.pyplot(fig_r)

            # --- Trend Analysis ---
            if "Date_dt" in comments.columns and not comments["Date_dt"].isna().all():
                st.subheader("📈 Trend Analysis")
                monthly_hotel = h_ratings.dropna(subset=["Date_dt"]).groupby(pd.Grouper(key="Date_dt", freq="M"))["Score"].mean()
                monthly_global = comments.dropna(subset=["Date_dt"]).groupby(pd.Grouper(key="Date_dt", freq="M"))["Score"].mean()

                fig_cmp, ax_cmp = plt.subplots()
                monthly_global.plot(ax=ax_cmp, marker="o", label="Global Avg", color="gray")
                monthly_hotel.plot(ax=ax_cmp, marker="o", label="This Hotel", color="blue")
                ax_cmp.set_ylabel("Avg Score"); ax_cmp.set_title("Hotel vs Global Trend")
                ax_cmp.legend(); st.pyplot(fig_cmp)

                # Compare with City (if available)
                if "City" in info.columns:
                    attr_value = info.loc[info["Hotel_ID"].astype(str) == hotel_id, "City"].values
                    if len(attr_value) > 0:
                        city = attr_value[0]
                        city_hotels = info[info["City"] == city]["Hotel_ID"].astype(str)
                        monthly_city = comments[comments["Hotel ID"].astype(str).isin(city_hotels)] \
                            .dropna(subset=["Date_dt"]).groupby(pd.Grouper(key="Date_dt", freq="M"))["Score"].mean()

                        fig_city, ax_city = plt.subplots()
                        monthly_global.plot(ax=ax_city, marker="o", label="Global Avg", color="gray")
                        monthly_city.plot(ax=ax_city, marker="o", label=f"{city} Avg", color="green")
                        monthly_hotel.plot(ax=ax_city, marker="o", label="This Hotel", color="blue")
                        ax_city.legend()
                        ax_city.set_title("Hotel vs City vs Global Trend")
                        st.pyplot(fig_city)

            # --- Text Insights ---
            if "Body" in h_ratings.columns:
                st.subheader("💬 Text Insights")
                h_ratings["Review_clean"] = h_ratings["Body"].apply(lambda x: clean_text(x))
                h_ratings["Review_clean"] = h_ratings["Review_clean"].fillna("")
                if "Review" in h_ratings.columns:
                    texts = h_ratings["Review"].fillna("")
                elif "Body" in h_ratings.columns:
                    texts = h_ratings["Body"].fillna("")
                elif "Comment" in h_ratings.columns:
                    texts = h_ratings["Comment"].fillna("")
                else:
                    texts = pd.Series("", index=h_ratings.index)

                pos = texts[h_ratings["Score_clean"] >= 8]
                neg = texts[h_ratings["Score_clean"] <= 7.5]

                pos_text = " ".join(pos.tolist()); neg_text = " ".join(neg.tolist())

                pos_counts = Counter(pos_text.split()); neg_counts = Counter(neg_text.split())
                top_pos = pos_counts.most_common(10); top_neg = neg_counts.most_common(10)

                cpos, cneg = st.columns(2)
                with cpos:
                    st.markdown("**Top 10 Positive Words**")
                    st.dataframe(pd.DataFrame(top_pos, columns=["Word", "Count"]))
                    if pos_text.strip():
                        wc = WordCloud(width=500, height=320, background_color="white", colormap="Blues").generate(pos_text)
                        fig_wc, ax_wc = plt.subplots(); ax_wc.imshow(wc); ax_wc.axis("off"); st.pyplot(fig_wc)
                with cneg:
                    st.markdown("**Top 10 Negative Words**")
                    st.dataframe(pd.DataFrame(top_neg, columns=["Word", "Count"]))
                    if neg_text.strip():
                        wc2 = WordCloud(width=500, height=320, background_color="white", colormap="Reds").generate(neg_text)
                        fig_wc2, ax_wc2 = plt.subplots(); ax_wc2.imshow(wc2); ax_wc2.axis("off"); st.pyplot(fig_wc2)

        else:
            st.warning("No hotels matched your search criteria.")
    st.divider()
    # ---------- Global ----------
    st.subheader("Global Comments Analysis")
    if "Body" in comments.columns:
        comments["clean_text"] = comments["Body"].apply(lambda x: clean_text(x, CUSTOM_STOPWORDS))
        pos_g = comments[comments["Score_clean"] >= 8]["clean_text"]
        neg_g = comments[comments["Score_clean"] <= 7]["clean_text"]
        pos_text_g = " ".join(pos_g.tolist()); neg_text_g = " ".join(neg_g.tolist())

        colG1, colG2 = st.columns(2)
        with colG1:
            st.markdown("**Positive WordCloud (Global)**")
            if pos_text_g.strip():
                wcg = WordCloud(width=600, height=350, background_color="white", colormap="Blues").generate(pos_text_g)
                fig_g1, ax_g1 = plt.subplots(); ax_g1.imshow(wcg); ax_g1.axis("off"); st.pyplot(fig_g1)
            # top 10
            top_pos_g = Counter(pos_text_g.split()).most_common(10)
            st.dataframe(pd.DataFrame(top_pos_g, columns=["Word", "Count"]))
        with colG2:
            st.markdown("**Negative WordCloud (Global)**")
            if neg_text_g.strip():
                wcg2 = WordCloud(width=600, height=350, background_color="white", colormap="Reds").generate(neg_text_g)
                fig_g2, ax_g2 = plt.subplots(); ax_g2.imshow(wcg2); ax_g2.axis("off"); st.pyplot(fig_g2)
            top_neg_g = Counter(neg_text_g.split()).most_common(10)
            st.dataframe(pd.DataFrame(top_neg_g, columns=["Word", "Count"]))

        # Score distribution & global trend
        fig_d, ax_d = plt.subplots()
        comments["Score"].hist(bins=10, edgecolor="black", ax=ax_d)
        ax_d.set_xlabel("Score"); ax_d.set_ylabel("Count"); ax_d.set_title("Score Distribution (Global)")
        st.pyplot(fig_d)

        if "Date_dt" in comments.columns and not comments["Date_dt"].isna().all():
            monthly_global = comments.dropna(subset=["Date_dt"]).groupby(pd.Grouper(key="Date_dt", freq="M"))["Score"].mean()
            fig_t, ax_t = plt.subplots()
            monthly_global.plot(ax=ax_t, marker="o")
            ax_t.set_ylabel("Avg Score"); ax_t.set_title("Average Score Over Time (Global)")
            st.pyplot(fig_t)
    else:
        st.warning("No `Comment` column in comments file — text analysis skipped.")

elif menu == "Recommendation / New Prediction":
    st.header("🏨 Recommendation Center")
    tabs = st.tabs([
        "🔎 By Query",
        "👤 By User (CF)",
        "📍 Similar Hotels",
        "⚖️ Hybrid (CF + CB)"
    ])
    # Shared filters
    # with st.sidebar.expander("⚙️ Bộ lọc chung", expanded=True):
    #     city_values = sorted(info["City"].dropna().unique()) if "City" in info.columns else []
    #     selected_city_cb = st.selectbox("Chọn thành phố", ["Tất cả"] + city_values, key="city_cb")

    #     star_col_guess = next((c for c in info.columns if "star" in c.lower() or "class" in c.lower()), None)
    #     column_cb = st.selectbox("Cột (nếu cần)", [None] + info.columns.tolist(),
    #                                     index=(info.columns.get_loc(star_col_guess) if star_col_guess else 0),
    #                                     key="column_cb")
    #     column_values = sorted(info[column_cb].dropna().unique()) if column_cb and column_cb in info.columns else []
    #     selected_col_cb = st.multiselect("Chọn giá trị", options=column_values, default=column_values, key="column_cb")

    # ==========================
    # TAB 0: Content-based by Query
    # ==========================
    with tabs[0]:
        query = st.text_area("Mô tả ưu tiên", value="", height=100, key="query_cb")
        topk_cb = st.number_input("Top K", min_value=5, max_value=50, value=10, step=1, key="topk_cb")
        min_reviews_cb = st.number_input("Số review tối thiểu", 0, 1000, 0, 10, key="min_reviews_cb")
        score_floor_cb = st.slider("Điểm review tối thiểu", 0.0, 10.0, 0.0, 0.5, key="score_cb")

        if st.button("Recommend (CB)", key="btn_cb"):
            out = cb_scores_from_text(query or "", top_k=int(topk_cb))
            # ... (same filter + display logic as before)
            

    # ==========================
    # TAB 1: Collaborative Filtering by User
    # ==========================
    with tabs[1]:
        if "Reviewer Label" in comments.columns:
            reviewer_label = st.selectbox("Chọn Reviewer Label", comments["Reviewer Label"].unique(), key="reviewer_cf")
            topk_cf = st.number_input("Top K", min_value=5, max_value=50, value=10, step=1, key="topk_cf")
            min_reviews_cf = st.number_input("Số review tối thiểu", 0, 1000, 0, 10, key="min_reviews_cf")
            score_floor_cf = st.slider("Điểm review tối thiểu", 0.0, 10.0, 0.0, 0.5, key="score_cf")

            if st.button("Recommend (CF)", key="btn_cf"):
                out = cf_recommend_for_user(reviewer_label, top_k=int(topk_cf))
                # ... (same filter + display logic as before)
        else:
            st.info("Không có Reviewer Label trong dữ liệu.")


    # ==========================
    # TAB 2: Similar Hotels
    # ==========================
    with tabs[2]:
        hotel_name_to_id = dict(zip(hotels_text["Hotel_Name"], hotels_text["Hotel_ID"]))
        hotel_selected_name = st.selectbox("Chọn khách sạn", hotels_text["Hotel_Name"].dropna().tolist(), key="hotel_sim")
        hotel_selected_id = hotel_name_to_id[hotel_selected_name]

        topk_sim = st.number_input("Top K gợi ý", min_value=3, max_value=50, value=10, step=1, key="topk_sim")
        min_reviews_sim = st.number_input("Số review tối thiểu", 0, 1000, 0, 10, key="min_reviews_sim")
        score_floor_sim = st.slider("Điểm review tối thiểu", 0.0, 10.0, 0.0, 0.5, key="score_sim")

        if st.button("Gợi ý khách sạn tương tự", key="btn_sim"):
            recs = recommend_hotels(hotel_selected_id, top_k=int(topk_sim))
            # ... (same filter + display logic as before)


    # ==========================
    # Hybrid Recommendation Function
    # ==========================
    def hybrid_recommend_for_user(reviewer_label: str, alpha: float = 0.5, top_k: int = 10) -> pd.DataFrame:
        """
        Combine CF and CB scores for a user.
        alpha: weight for CF (0 = only CB, 1 = only CF)
        """
        # Get CF recommendations
        cf_df = cf_recommend_for_user(reviewer_label, top_k=top_k * 2)
        # Get CB recommendations based on user's reviews
        user_reviews = comments[comments["Reviewer Label"] == reviewer_label]
        if not user_reviews.empty:
            cb_query = " ".join(user_reviews["Body"].fillna("").tolist())
            cb_df = cb_scores_from_text(cb_query, top_k=top_k * 2)
        else:
            cb_df = pd.DataFrame()
        # Merge CF and CB
        if not cf_df.empty and not cb_df.empty:
            merged = pd.merge(cf_df, cb_df, on=["Hotel_ID"], how="outer")
            merged["cf_score"] = merged["cf_score"].fillna(0)
            merged["cb_score"] = merged["cb_score"].fillna(0)
            merged["hybrid_score"] = alpha * merged["cf_score"] + (1 - alpha) * merged["cb_score"]
            merged = merged.sort_values("hybrid_score", ascending=False).head(top_k)
            return merged[["Hotel_ID", "Hotel_Name", "Hotel_Address", "hybrid_score"]]
        elif not cf_df.empty:
            cf_df["hybrid_score"] = cf_df["cf_score"]
            return cf_df[["Hotel_ID", "Hotel_Name", "Hotel_Address", "hybrid_score"]].head(top_k)
        elif not cb_df.empty:
            cb_df["hybrid_score"] = cb_df["cb_score"]
            return cb_df[["Hotel_ID", "Hotel_Name", "Hotel_Address", "hybrid_score"]].head(top_k)
        else:
            return pd.DataFrame({"note": ["No recommendations available."]})

    # ==========================
    # TAB 3: Hybrid (CF + CB)
    # ==========================
    with tabs[3]:
        if "Reviewer Label" in comments.columns:
            reviewer_label_hybrid = st.selectbox("Chọn Reviewer Label (Hybrid)", comments["Reviewer Label"].unique(), key="reviewer_hybrid")
            topk_hybrid = st.number_input("Top K", min_value=5, max_value=50, value=10, step=1, key="topk_hybrid")
            alpha = st.slider("Trọng số α (CF share)", 0.0, 1.0, 0.5, 0.05, key="alpha_hybrid")
            min_reviews_hybrid = st.number_input("Số review tối thiểu", 0, 1000, 0, 10, key="min_reviews_hybrid")
            score_floor_hybrid = st.slider("Điểm review tối thiểu", 0.0, 10.0, 0.0, 0.5, key="score_hybrid")

            if st.button("Recommend (Hybrid)", key="btn_hybrid"):
                out = hybrid_recommend_for_user(reviewer_label_hybrid, alpha=alpha, top_k=int(topk_hybrid))
                # ... (same filter + display logic as before)
        else:
            st.info("Không có Reviewer Label trong dữ liệu.")
