# app.py
# Streamlit: Recommender cho Agoda – Content-based (gensim) + Hybrid CF item-item
# Menu: Introduction, Business Problem, Evaluation & Report, New Prediction / Analysis, Recommendation

import streamlit as st
import pandas as pd
import numpy as np
import re, io, base64
from typing import List, Tuple, Dict, Optional
from sklearn.feature_extraction.text import TfidfVectorizer

# --------- Optional: sklearn cosine fallback & charts ---------
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Hotel Recommender (Agoda)", layout="wide")


# ==========================
# Helpers
# ==========================
def clean_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace(",", ".", regex=False), errors="coerce")

def load_vi_stopwords(file: Optional[io.BytesIO]) -> set:
    sw = set()
    if file is None:
        # Fallback VN stopwords (rút gọn) – bạn có thể upload file đầy đủ ở sidebar
        core = """
        và nhưng hoặc vì thì là của những các cho với từ một một số khi nơi nữa lại đã đang sẽ
        tại trên dưới trong ngoài về như vậy nên do bởi nếu thì được tới đến đến từ
        """
        for w in core.split():
            sw.add(w.strip().lower())
        return sw
    txt = file.read().decode("utf-8", errors="ignore")
    for line in txt.splitlines():
        line = line.strip().lower()
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

def tokenize(text, vi_sw: set, en_sw: set = EN_SW) -> List[str]:
    if isinstance(text, list):
        text = " ".join(map(str, text))
    elif not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r"[^a-zA-ZÀ-ỹ\s]", " ", text)
    toks = [t for t in text.split() if t and (t not in vi_sw) and (t not in en_sw)]
    return [t for t in toks if len(t) >= 2]


def embed_pdf(bts: bytes, height=900):
    b64 = base64.b64encode(bts).decode("utf-8")
    html = f'<iframe src="data:application/pdf;base64,{b64}" width="100%" height="{height}px" type="application/pdf"></iframe>'
    st.components.v1.html(html, height=height+10, scrolling=True)

def embed_html(bts: bytes, height=800):
    html = bts.decode("utf-8", errors="ignore")
    st.components.v1.html(html, height=height, scrolling=True)

# ==========================
# Sidebar – Data & Settings
# ==========================
st.sidebar.header("⚙️ Cấu hình")
uploaded_comments = st.sidebar.file_uploader("Upload hotel_comments.csv (tùy chọn)", type=["csv"])
uploaded_info     = st.sidebar.file_uploader("Upload hotel_info.csv (tùy chọn)", type=["csv"])
uploaded_vi_sw    = st.sidebar.file_uploader("Upload vietnamese-stopwords.txt (tùy chọn)", type=["txt"])

@st.cache_data(show_spinner=False)
def load_datasets(u_comments, u_info) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if u_comments is not None:
        comments = pd.read_csv(u_comments)
    else:
        comments = pd.read_csv("sample_hotel_comments.csv")
    if u_info is not None:
        info = pd.read_csv(u_info)
    else:
        info = pd.read_csv("hotel_info.csv")

    # Clean basics
    comments["Score_clean"] = clean_numeric(comments.get("Score", pd.Series(index=comments.index))).clip(lower=0, upper=10)
    if "Review Date" in comments.columns:
        comments["Review Date"] = pd.to_datetime(comments["Review Date"], errors="coerce")

    for col in ["Total_Score","Location","Cleanliness","Service","Facilities","Value_for_money","Comfort_and_room_quality"]:
        if col in info.columns:
            info[col + "_clean"] = clean_numeric(info[col]).clip(lower=0, upper=10)

    # Ensure IDs are strings
    if "Hotel ID" in comments.columns:
        comments["Hotel ID"] = comments["Hotel ID"].astype(str)
    if "Reviewer ID" in comments.columns:
        comments["Reviewer ID"] = comments["Reviewer ID"].astype(str)
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
    # Prepare text: description + aggregated review titles/bodies
    info_text = info[["Hotel_ID","Hotel_Name","Hotel_Address","Hotel_Description"]].copy()
    info_text["Hotel_Description"] = info_text["Hotel_Description"].fillna("") if "Hotel_Description" in info_text else ""

    if "Title" in comments.columns and "Body" in comments.columns:
        reviews_text = (comments.groupby("Hotel ID")
                        .agg(titles=("Title", lambda x: " ".join(map(str, x.dropna().tolist()))[:200000]),
                             bodies=("Body",  lambda x: " ".join(map(str, x.dropna().tolist()))[:400000]))
                        .reset_index().rename(columns={"Hotel ID":"Hotel_ID"}))
    else:
        # If no review text, still return blank texts.
        reviews_text = comments.groupby("Hotel ID").size().reset_index().rename(columns={"Hotel ID":"Hotel_ID",0:"cnt"})
        reviews_text["titles"] = ""
        reviews_text["bodies"] = ""

    hotels_text = info_text.merge(reviews_text, on="Hotel_ID", how="left")
    for col in ["titles","bodies"]:
        if col in hotels_text:
            hotels_text[col] = hotels_text[col].fillna("")
    hotels_text["doc_raw"] = (
        hotels_text.get("Hotel_Description","").astype(str) + " " +
        hotels_text.get("titles","").astype(str) + " " +
        hotels_text.get("bodies","").astype(str)
    )
    tokens_list = [tokenize(t, vi_sw) for t in hotels_text["doc_raw"].tolist()]
    hotels_text["doc"] = [" ".join(t) for t in tokens_list]
    return hotels_text, tokens_list

hotels_text, tokens_list = build_hotel_docs(comments, info, VI_SW)

# ==========================
# Content-based (gensim)
# ==========================
@st.cache_resource(show_spinner=False)
def build_cb_index(docs):
    vectorizer = TfidfVectorizer(max_features=10000)
    tfidf_matrix = vectorizer.fit_transform(docs)
    return vectorizer, tfidf_matrix

vectorizer, tfidf_matrix = build_cb_index(hotels_text["doc"].tolist())


dictionary, tfidf, cb_index = build_cb_index(tokens_list)

# ==========================
# Similarity matrix (gensim)

def build_similarity_matrix():
    sims_matrix = cosine_similarity(tfidf_matrix)
    df_sims = pd.DataFrame(sims_matrix,
                           index=hotels_text["Hotel_ID"],
                           columns=hotels_text["Hotel_ID"])
    return df_sims

df_sims = build_similarity_matrix()


df_sims = build_similarity_matrix(tokens_list, hotels_text)

def recommend_hotels(hotel_id: str, sims_df: pd.DataFrame, hotels_df: pd.DataFrame, top_k: int = 5):
    if hotel_id not in sims_df.index:
        return pd.DataFrame({"note": [f"{hotel_id} không có trong similarity matrix"]})

    sims = sims_df.loc[hotel_id].drop(hotel_id)
    top_ids = sims.sort_values(ascending=False).head(top_k).index

    recs = hotels_df[hotels_df["Hotel_ID"].isin(top_ids)][["Hotel_ID","Hotel_Name","Hotel_Address"]]
    recs = recs.set_index("Hotel_ID").loc[top_ids].reset_index()
    recs["similarity"] = sims.loc[top_ids].values
    return recs


# ==========================
def cb_scores_from_text(query: str, top_k=20) -> pd.DataFrame:
    q_vec = vectorizer.transform([query])
    sims = cosine_similarity(q_vec, tfidf_matrix).flatten()
    order = sims.argsort()[::-1][:top_k]
    rows = []
    for j in order:
        rows.append({
            "Hotel_ID": hotels_text.iloc[j]["Hotel_ID"],
            "Hotel_Name": hotels_text.iloc[j]["Hotel_Name"],
            "Hotel_Address": hotels_text.iloc[j].get("Hotel_Address",""),
            "cb_score": float(sims[j])
        })
    return pd.DataFrame(rows)


def cb_scores_for_user(reviewer_id: str, score_floor: float = 8.5, top_k=20) -> pd.DataFrame:
    ur = comments[comments["Reviewer ID"] == reviewer_id].dropna(subset=["Score_clean"])
    if ur.empty:
        return pd.DataFrame()
    liked_hotels = ur[ur["Score_clean"] >= score_floor]["Hotel ID"].tolist()
    if not liked_hotels:
        return pd.DataFrame()

    liked_idx = hotels_text.index[hotels_text["Hotel_ID"].isin(liked_hotels)].tolist()
    if not liked_idx:
        return pd.DataFrame()

    # Average vector of liked hotels
    user_vec = tfidf_matrix[liked_idx].mean(axis=0)
    sims = cosine_similarity(user_vec, tfidf_matrix).flatten()

    seen = set(liked_hotels)
    order = sims.argsort()[::-1]
    out = []
    for j in order:
        hid = hotels_text.iloc[j]["Hotel_ID"]
        if hid in seen: 
            continue
        out.append({
            "Hotel_ID": hid,
            "Hotel_Name": hotels_text.iloc[j]["Hotel_Name"],
            "Hotel_Address": hotels_text.iloc[j].get("Hotel_Address",""),
            "cb_score": float(sims[j])
        })
        if len(out) >= top_k:
            break
    return pd.DataFrame(out)


# ==========================
# Collaborative (item–item)
# ==========================
@st.cache_resource(show_spinner=False)
def build_item_item(comments: pd.DataFrame):
    # Build user-item (mean) and mean-center by user
    if not {"Reviewer ID","Hotel ID","Score_clean"}.issubset(comments.columns):
        return None
    ui = comments.pivot_table(index="Reviewer ID", columns="Hotel ID", values="Score_clean", aggfunc="mean")
    if ui.empty:
        return None
    user_mean = ui.mean(axis=1)
    ui_centered = ui.sub(user_mean, axis=0)
    X = ui_centered.fillna(0).to_numpy(dtype=float)  # users x items
    norms = np.linalg.norm(X, axis=0, keepdims=True)
    norms[norms==0] = 1.0
    Xn = X / norms
    item_sim = Xn.T @ Xn   # items x items
    return {
        "ui": ui,
        "ui_centered": ui_centered,
        "item_sim": item_sim,
        "item_ids": ui.columns.tolist(),
        "user_mean": user_mean
    }

cf_state = build_item_item(comments)

def cf_recommend_for_user(reviewer_id: str, top_k=20) -> pd.DataFrame:
    if cf_state is None or reviewer_id not in cf_state["ui"].index:
        return pd.DataFrame()
    ui, ui_centered, item_sim, item_ids, user_mean = (
        cf_state["ui"], cf_state["ui_centered"], cf_state["item_sim"], cf_state["item_ids"], cf_state["user_mean"]
    )
    seen_mask = ui.loc[reviewer_id].notna().values
    preds = item_sim @ ui_centered.loc[reviewer_id].fillna(0).to_numpy(dtype=float)
    preds[seen_mask] = -1e9
    top_idx = np.argsort(-preds)[:top_k]
    id_name = info[["Hotel_ID","Hotel_Name","Hotel_Address"]].drop_duplicates()
    df = pd.DataFrame({"Hotel_ID":[item_ids[j] for j in top_idx],
                       "cf_score": [float(preds[j]) for j in top_idx]})
    df = df.merge(id_name, left_on="Hotel_ID", right_on="Hotel_ID", how="left")
    return df

# ==========================
# Hybrid (blend CB + CF)
# ==========================
def hybrid_for_user(reviewer_id: str, alpha: float = 0.8, top_k=20) -> pd.DataFrame:
    cb = cb_scores_for_user(reviewer_id, top_k=100)
    cf = cf_recommend_for_user(reviewer_id, top_k=100)
    if cb.empty and cf.empty:
        return pd.DataFrame()
    # Normalize to 0..1 per method
    def norm01(x):
        if x.empty or x.max() == x.min():
            return pd.Series(np.zeros(len(x)), index=x.index)
        return (x - x.min()) / (x.max() - x.min())

    df = pd.merge(cb, cf[["Hotel_ID","cf_score"]], on="Hotel_ID", how="outer")
    df["cb_score"] = df["cb_score"].fillna(0.0)
    df["cf_score"] = df["cf_score"].fillna(0.0)
    df["cb_n"] = norm01(df["cb_score"])
    df["cf_n"] = norm01(df["cf_score"])
    df["hybrid"] = alpha * df["cb_n"] + (1-alpha) * df["cf_n"]
    df = df.merge(hotels_text[["Hotel_ID","Hotel_Name","Hotel_Address"]], on="Hotel_ID", how="left").drop_duplicates("Hotel_ID")
    df = df.sort_values("hybrid", ascending=False).head(top_k)
    return df[["Hotel_ID","Hotel_Name","Hotel_Address","cb_score","cf_score","hybrid"]]

def hybrid_from_query(query: str, alpha: float = 1.0, top_k=20) -> pd.DataFrame:
    # alpha=1.0 → thuần CB cho text query
    cb = cb_scores_from_text(query, top_k=200)
    # CF không có ngữ cảnh user, để 0
    cb["cf_score"] = 0.0
    cb["hybrid"] = cb["cb_score"]  # alpha=1
    return cb.head(top_k)

hotel_review_counts = comments.groupby("Hotel ID").size().sort_values(ascending=False).head(20)
top_nat = comments["Nationality"].value_counts().head(10).index
subset = comments[comments["Nationality"].isin(top_nat)]


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
    st.title("Hotel Recommendation – Agoda (Content-based + Hybrid CF)")
    st.markdown("""
**Mục tiêu**: Gợi ý khách sạn theo **nội dung mô tả + đánh giá** (gensim TF-IDF + cosine), kết hợp **CF item-item** khi có chồng chéo người dùng, và cung cấp EDA/tương tác dự đoán cho dữ liệu mới.

**Dữ liệu**:
- `hotel_comments.csv`: đánh giá/điểm/tiêu đề/nội dung theo người dùng – khách sạn.
- `hotel_info.csv`: thông tin khách sạn, mô tả, địa chỉ, các tiêu chí (Location, Cleanliness, ...).
- (Tùy chọn) `vietnamese-stopwords.txt`: stopwords tiếng Việt để làm sạch văn bản.

Bạn có thể upload thay thế file ở **Sidebar**.
    """)
    st.image("Agoda_img.jpg",
         caption="Agoda – Online Travel Booking", use_container_width=True)

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
    st.image("hotel_review.avif", caption="Hotel Reviews", use_container_width=True)
# ==========================
# 3) Evaluation & Report
# ==========================
elif menu == "Evaluation & Report":
    st.header("Evaluation & Report")


    # ---- EDA auto from CSV ----
    st.subheader("EDA (tự động từ CSV)")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Số review", f"{len(comments):,}")
        st.metric("Số khách sạn", f"{info['Hotel_ID'].nunique():,}")
        if "Reviewer ID" in comments:
            st.metric("Số người dùng", f"{comments['Reviewer ID'].nunique():,}")
    with col2:
        st.write("Phân phối điểm review (Score_clean)")
        sc = comments["Score_clean"].dropna()
        st.bar_chart(pd.DataFrame({"count": sc.value_counts().sort_index()}))

    # ---- Extra chart: số lượng review theo khách sạn ----
    st.subheader("Top khách sạn theo số lượng review")

    st.bar_chart(hotel_review_counts)

    # ---- Extra chart: boxplot theo quốc tịch ----
    if "Nationality" in comments.columns:
        st.subheader("Boxplot điểm theo quốc tịch (top 10)")
        
        import seaborn as sns, matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10,5))
        sns.boxplot(x="Nationality", y="Score_clean", data=subset, ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        st.pyplot(fig)

    # ---- Correlation with Total Score ----
    st.write("Tương quan tiêu chí với Total_Score (nếu có):")
    sub_cols = [c for c in ["Location_clean","Cleanliness_clean","Service_clean","Facilities_clean",
                            "Value_for_money_clean","Comfort_and_room_quality_clean"] if c in info.columns]
    if sub_cols and "Total_Score_clean" in info.columns:
        corr = info[["Total_Score_clean"] + sub_cols].corr(numeric_only=True)["Total_Score_clean"].drop("Total_Score_clean").sort_values(ascending=False)
        st.dataframe(corr.to_frame("corr_with_total"))

        # Extra heatmap
        st.subheader("Heatmap tương quan các tiêu chí")
        corrmat = info[["Total_Score_clean"] + sub_cols].corr()
        fig, ax = plt.subplots(figsize=(8,6))
        sns.heatmap(corrmat, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        st.pyplot(fig)

        # Extra scatter plot
        if "Location_clean" in info.columns:
            st.subheader("Scatter plot: Location vs Total Score")
            fig, ax = plt.subplots()
            ax.scatter(info["Location_clean"], info["Total_Score_clean"], alpha=0.5)
            ax.set_xlabel("Location")
            ax.set_ylabel("Total Score")
            st.pyplot(fig)

    st.divider()

    # ---- Upload external EDA reports ----
    st.subheader("EDA từ PDF (nếu bạn có)")
    pdf1 = st.file_uploader("Upload Hotel Comments EDA.pdf", type=["pdf"], key="pdf_comments")
    pdf2 = st.file_uploader("Upload Hotel Info EDA.pdf", type=["pdf"], key="pdf_info")
    colp1, colp2 = st.columns(2)
    with colp1:
        if pdf1 is not None:
            st.caption("Hotel Comments EDA.pdf")
            embed_pdf(pdf1.read(), height=600)
    with colp2:
        if pdf2 is not None:
            st.caption("Hotel Info EDA.pdf")
            embed_pdf(pdf2.read(), height=600)

    # ---- NEW: Upload HTML reports ----
    st.subheader("EDA từ HTML (nếu bạn có)")
    html1 = st.file_uploader("Upload Hotel Comments EDA.html", type=["html"], key="html_comments")
    html2 = st.file_uploader("Upload Hotel Info EDA.html", type=["html_info"])
    def embed_html(bts: bytes, height=800):
        html = bts.decode("utf-8", errors="ignore")
        st.components.v1.html(html, height=height, scrolling=True)
    colh1, colh2 = st.columns(2)
    with colh1:
        if html1 is not None:
            st.caption("Hotel Comments EDA.html")
            embed_html(html1.read(), height=600)
    with colh2:
        if html2 is not None:
            st.caption("Hotel Info EDA.html")
            embed_html(html2.read(), height=600)

    st.divider()

    # ---- Modeling summary ----
    st.subheader("Modeling")
    st.markdown("""
- **Content-based**: Gensim TF-IDF + cosine từ **mô tả + tiêu đề + nội dung review** theo khách sạn.  
- **Collaborative (item–item)**: dựa trên ma trận user-item mean-centered; dùng khi có chồng chéo.  
- **Hybrid**: trộn điểm (mặc định **80% CB + 20% CF**; có thể chỉnh).
    """)
    st.markdown("**Lưu ý**: do dữ liệu đánh giá thưa, nên ưu tiên CB (mô tả + review) để gợi ý khách sạn mới hoặc không có chồng chéo người dùng.")
    st.markdown("""
**Đánh giá**: do không có tập test, nên đánh giá sơ bộ bằng cách xem xét thủ công kết quả gợi ý (qualitative).  
- Với truy vấn văn bản, kết quả gợi ý khá hợp lý (xem tab New Prediction / Analysis).  
- Với người dùng có chồng chéo, kết quả gợi ý cũng khá hợp lý (xem tab New Prediction / Analysis).  
- Với người dùng không có chồng chéo, kết quả gợi ý chỉ dựa trên CB.
    """)

# ==========================
# 4) Insight (per-hotel with map)
# ==========================
elif menu == "Insight":
    st.header("📊 Insight cho từng khách sạn")

    # --- Hotel selection ---
    hotel_name_to_id = dict(zip(info["Hotel_Name"], info["Hotel_ID"]))
    selected_hotel_name = st.selectbox("Chọn khách sạn", sorted(info["Hotel_Name"].dropna().unique().tolist()))
    selected_hotel_id = hotel_name_to_id[selected_hotel_name]

    st.markdown(f"### 🏨 {selected_hotel_name}")
    st.caption(f"Hotel ID: {selected_hotel_id}")

    # --- Filter comments for this hotel ---
    hotel_reviews = comments[comments["Hotel ID"] == selected_hotel_id].copy()
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
    if "Review Date" in hotel_reviews.columns:
        hotel_reviews["year_month"] = hotel_reviews["Review Date"].dt.to_period("M").astype(str)
        trend = hotel_reviews.groupby("year_month")["Score_clean"].agg(["count","mean"]).reset_index()

        st.subheader("📈 Xu hướng review & điểm trung bình theo tháng")
        import matplotlib.pyplot as plt
        fig, ax1 = plt.subplots(figsize=(12,5))
        ax2 = ax1.twinx()
        ax1.bar(trend["year_month"], trend["count"], color="skyblue", alpha=0.6)
        ax2.plot(trend["year_month"], trend["mean"], color="red", marker="o")
        ax1.set_ylabel("Số review")
        ax2.set_ylabel("Điểm trung bình")
        ax1.set_xticks(range(0,len(trend), max(1,len(trend)//12)))
        ax1.set_xticklabels(trend["year_month"][::max(1,len(trend)//12)], rotation=45, ha="right")
        st.pyplot(fig)

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
        wc = WordCloud(width=800, height=400, background_color="white").generate(text)
        fig, ax = plt.subplots(figsize=(10,5))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)
    pos_texts = (hotel_reviews["Title"].fillna("") + " " + hotel_reviews["Body"].fillna("")).loc[hotel_reviews["Score_clean"]>=9].tolist()
    neg_texts = (hotel_reviews["Title"].fillna("") + " " + hotel_reviews["Body"].fillna("")).loc[hotel_reviews["Score_clean"]<=7].tolist()
    make_wc(pos_texts, "Review tích cực (≥9)")
    make_wc(neg_texts, "Review tiêu cực (≤7)")

    # --- Recommendations for improvement ---
    st.subheader("📌 Gợi ý cho khách sạn này")
    st.markdown(f"""
- Tổng quan: khách sạn **{selected_hotel_name}** có {len(hotel_reviews):,} review với điểm trung bình **{hotel_reviews['Score_clean'].mean():.2f}**.  
- **Quốc tịch nổi bật**: {', '.join(hotel_reviews['Nationality'].value_counts().head(3).index.tolist()) if 'Nationality' in hotel_reviews else 'N/A'}.  
- **Nhóm khách chính**: {grp_counts.index[0] if ('Group' in hotel_reviews or 'Traveler type' in hotel_reviews) else 'N/A'}.  
- Gợi ý cải thiện: tập trung xử lý các yếu tố hay bị phàn nàn trong WordCloud tiêu cực (wifi, ồn ào, dịch vụ chậm...).  
- Marketing: khai thác điểm mạnh từ WordCloud tích cực (view đẹp, sạch sẽ, vị trí tốt) để quảng bá.
    """)

# ==========================
# Recommendation 
# ==========================
elif menu == "Recommendation / New Prediction":
    st.header("🏨 Recommendation Center")


    tabs = st.tabs([
        "🔎 By Query / User",
        "🏨 For New Hotels",
        "🎯 By Filters",
        "📍 Similar Hotels"
    ])

    # Helper: safe merge with hotel info
    def merge_with_info(df):
        cols_needed = ["Hotel_ID","Hotel_Name","Hotel_Address"]
        if all(c in info.columns for c in cols_needed):
            return df.merge(info[cols_needed], on="Hotel_ID", how="left")
        elif "Hotel_ID" in info.columns:
            return df.merge(info[["Hotel_ID"]], on="Hotel_ID", how="left")
        else:
            return df

    # Helper: safe display
    def safe_display(df, cols):
        cols_present = [c for c in cols if c in df.columns]
        st.dataframe(df[cols_present])

    # ---- TAB 1: By Query/User ----
    with tabs[0]:
        st.subheader("Gợi ý theo truy vấn văn bản hoặc sở thích người dùng")
        query = st.text_area("Mô tả ưu tiên", value="", height=100)
        alpha = st.slider("Trọng số CB trong Hybrid (alpha)", 0.0, 1.0, 0.8, 0.05, key="alpha_query")
        topk = st.number_input("Top K", 5, 50, 10, key="topk_query")
        use_user = st.checkbox("Kết hợp sở thích từ người dùng", value=False, key="cb_user_check")
        reviewer_id = ""
        if use_user and "Reviewer ID" in comments.columns:
            reviewer_id = st.text_input("Reviewer ID", value=comments["Reviewer ID"].iloc[0])

        if st.button("Recommend", key="btn_query"):
            if use_user and reviewer_id:
                df_user = cb_scores_for_user(reviewer_id, top_k=200)
                df_query = cb_scores_from_text(query or "", top_k=200)
                def norm01(col):
                    return (col - col.min())/(col.max()-col.min()) if col.max()!=col.min() else col*0
                m = pd.merge(df_user, df_query, on=["Hotel_ID"], how="outer").fillna(0)
                m["cb_score"] = 0.5*norm01(m["cb_score_x"]) + 0.5*norm01(m["cb_score_y"])
                cf = cf_recommend_for_user(reviewer_id, top_k=200)
                m = m.merge(cf[["Hotel_ID","cf_score"]], on="Hotel_ID", how="left").fillna({"cf_score":0.0})
                m["hybrid"] = alpha*norm01(m["cb_score"]) + (1-alpha)*norm01(m["cf_score"])
                out = merge_with_info(m).sort_values("hybrid", ascending=False).head(int(topk))
                safe_display(out, ["Hotel_ID","Hotel_Name","Hotel_Address","cb_score","cf_score","hybrid"])
            else:
                out = hybrid_from_query(query or "", alpha=1.0, top_k=int(topk))
                out = merge_with_info(out)
                safe_display(out, ["Hotel_ID","Hotel_Name","Hotel_Address","cb_score","hybrid"])

    # ---- TAB 2: For New Hotels ----
    with tabs[1]:
        st.subheader("Gợi ý cho khách sạn mới")
        new_hotels = st.file_uploader("Upload new_hotels.csv", type=["csv"], key="up_new_hotels")
        if new_hotels is not None:
            df_new = pd.read_csv(new_hotels)
            df_new["doc_raw"] = df_new.get("Hotel_Description","").astype(str)
            df_new["doc"] = df_new["doc_raw"].apply(lambda t: " ".join(tokenize(t, VI_SW)))
            st.success(f"Nạp {len(df_new)} khách sạn mới.")
            if st.button("Tìm khách sạn tương tự", key="btn_new_hotel"):
                rows = []
                for _, r in df_new.iterrows():
                    q_toks = r["doc"].split() if isinstance(r["doc"], str) else []
                    q_bow  = dictionary.doc2bow(q_toks)
                    sims   = cb_index[tfidf[q_bow]] if len(q_bow)>0 else np.zeros(len(hotels_text))
                    best = np.argsort(-sims)[:10]
                    sim_to_existing = [(hotels_text.iloc[j]["Hotel_ID"], hotels_text.iloc[j]["Hotel_Name"], float(sims[j])) for j in best]
                    rows.append({
                        "New_Hotel_ID": r.get("Hotel_ID",""),
                        "New_Hotel_Name": r.get("Hotel_Name",""),
                        "Similar_existing": sim_to_existing[:5]
                    })
                st.dataframe(pd.DataFrame(rows))

    # ---- TAB 3: By Filters ----
    with tabs[2]:
        st.subheader("Gợi ý theo bộ lọc")
        colf = st.columns(3)
        with colf[0]:
            star_col_guess = None
            for c in info.columns:
                if "star" in c.lower() or "class" in c.lower():
                    star_col_guess = c
                    break
            star_col = st.selectbox("Cột 'sao' (nếu có)", [star_col_guess] + [c for c in info.columns if c!=star_col_guess])
            star_values = sorted(info[star_col].dropna().unique()) if star_col and star_col in info.columns else []
            selected_stars = st.multiselect("Chọn sao", options=star_values, default=star_values)
        with colf[1]:
            address_query = st.text_input("Địa chỉ chứa", "")
        with colf[2]:
            phrase = st.text_input("Cụm từ trong mô tả/đánh giá", "")

        min_reviews = st.number_input("Số review tối thiểu", 0, 1000, 0, 10)
        score_floor = st.slider("Điểm review tối thiểu", 0.0, 10.0, 0.0, 0.5, key="score_filters")
        alpha = st.slider("Trọng số CB trong hybrid", 0.0, 1.0, 0.8, 0.05, key="alpha_filters")
        topk  = st.number_input("Top K", 5, 50, 10, key="topk_filters")
        reviewer_id = st.text_input("Reviewer ID (tùy chọn, để tăng CF):",
                                    value="" if "Reviewer ID" not in comments else comments["Reviewer ID"].iloc[0])

        base = pd.DataFrame()
        if phrase.strip():
            base = cb_scores_from_text(phrase.strip(), top_k=500)
        elif reviewer_id:
            base = cb_scores_for_user(reviewer_id, top_k=500)
        else:
            st.info("Nhập **cụm từ** hoặc **Reviewer ID** để khởi tạo gợi ý.")
            st.stop()

        if reviewer_id:
            cf = cf_recommend_for_user(reviewer_id, top_k=500)
            base = base.merge(cf[["Hotel_ID","cf_score"]], on="Hotel_ID", how="left").fillna({"cf_score":0.0})
        else:
            base["cf_score"] = 0.0

        df = merge_with_info(base).drop_duplicates("Hotel_ID")

        # Apply filters
        if star_col in df.columns and selected_stars:
            df = df[df[star_col].isin(selected_stars)]
        if address_query.strip():
            df = df[df["Hotel_Address"].fillna("").str.contains(address_query.strip(), case=False, na=False)]
        if min_reviews > 0 and "Hotel ID" in comments.columns:
            review_counts = comments["Hotel ID"].value_counts()
            df = df[df["Hotel_ID"].isin(review_counts[review_counts >= min_reviews].index)]
        if score_floor > 0 and "Total_Score_clean" in df.columns:
            df = df[df["Total_Score_clean"].fillna(0) >= score_floor]

        def safe_norm(s): 
            s = s.fillna(0)
            return (s - s.min())/(s.max()-s.min()) if s.max()!=s.min() else s*0
        df["cb_n"] = safe_norm(df["cb_score"])
        df["cf_n"] = safe_norm(df["cf_score"])
        df["hybrid"] = alpha*df["cb_n"] + (1-alpha)*df["cf_n"]

        df = df.sort_values("hybrid", ascending=False).head(int(topk))
        safe_display(df, ["Hotel_ID","Hotel_Name","Hotel_Address","cb_score","cf_score","hybrid"])

    # ---- TAB 4: Similar Hotels ----
    with tabs[3]:
        st.subheader("Khách sạn tương tự")
        hotel_name_to_id = dict(zip(hotels_text["Hotel_Name"], hotels_text["Hotel_ID"]))
        hotel_selected_name = st.selectbox("Chọn Hotel_Name", hotels_text["Hotel_Name"].tolist())
        hotel_selected_id = hotel_name_to_id[hotel_selected_name]
        topk_sim = st.number_input("Top K gợi ý", 3, 20, 5, key="sim_topk")
        if st.button("Gợi ý khách sạn tương tự", key="btn_similar"):
            recs = recommend_hotels(hotel_selected_id, df_sims, hotels_text, top_k=int(topk_sim))
            recs = merge_with_info(recs)
            safe_display(recs, ["Hotel_ID","Hotel_Name","Hotel_Address","similarity"])
