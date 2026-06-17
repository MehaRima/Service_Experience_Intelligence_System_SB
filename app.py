import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from data_utils import generate_synthetic_experience_data, prepare_experience_data, add_sentiment, date_filter, recommendations

st.set_page_config(page_title="Service Experience Intelligence", page_icon="💬", layout="wide")
st.markdown("""
<style>
.block-container {padding-top:1rem;}
.panel {background:#fff7ed; color:#7c2d12; border:1px solid #fed7aa; padding:14px; border-radius:16px;}
.panel * {color:#7c2d12 !important;}
div[data-testid="stMetric"] {background:#ffffff; border:1px solid #fed7aa; border-radius:14px; padding:12px;}
div[data-testid="stMetric"] * {color:#431407 !important;}
</style>
""", unsafe_allow_html=True)
st.title("💬 Service Experience Intelligence System")
st.caption("Understand satisfaction, service quality, feedback themes, and improvement priorities.")

with st.sidebar:
    st.header("Feedback Source")
    up=st.file_uploader("Upload feedback CSV", type=["csv"])
    n=st.slider("Demo feedback rows", 500, 8000, 2500, step=500)
    if up:
        try: df=prepare_experience_data(pd.read_csv(up)); src="Uploaded CSV"
        except Exception as e: st.error(e); df=generate_synthetic_experience_data(n); src="Synthetic Demo"
    else:
        df=generate_synthetic_experience_data(n); src="Synthetic Demo"
    df=add_sentiment(df)
    st.success(src)
    period=st.selectbox("Feedback period", ["All Data","Last 30 Days","Last 90 Days","Last 180 Days"])
    df=date_filter(df, period)
    areas=["All Services"]+sorted(df["service_area"].dropna().unique().tolist())
    area=st.selectbox("Service area", areas)
    if area!="All Services": df=df[df["service_area"]==area]
    st.info(f"{len(df):,} feedback records")

if df.empty:
    st.warning("No feedback records in the selected view.")
    st.stop()

m1,m2,m3,m4=st.columns(4)
m1.metric("Feedback Records", f"{len(df):,}")
m2.metric("Avg Rating", f"{df['rating'].mean():.2f}/5")
m3.metric("Experience Score", f"{df['experience_score'].mean():.1f}")
m4.metric("High-Risk Share", f"{(df['risk_band'].astype(str)=='High Risk').mean()*100:.1f}%")

left,right=st.columns([.9,1.1])
with left:
    st.markdown('<div class="panel"><b>Experience Lens</b><br>The score blends rating, sentiment keywords, and response-time pressure.</div>', unsafe_allow_html=True)
    st.subheader("Risk Band Mix")
    fig=px.pie(df, names="risk_band", title="Experience Health Distribution")
    st.plotly_chart(fig, use_container_width=True)
with right:
    st.subheader("Service Experience by Area")
    area_sum=df.groupby("service_area").agg(records=("rating","count"), avg_rating=("rating","mean"), experience=("experience_score","mean")).reset_index()
    fig=px.bar(area_sum, x="service_area", y="experience", color="records", title="Average Experience Score by Service Area")
    st.plotly_chart(fig, use_container_width=True)

tab1,tab2,tab3=st.tabs(["Feedback Themes","Trend & Deterioration","Improvement Actions"])
with tab1:
    st.subheader("Text Theme Signals")
    texts=df["feedback_text"].fillna("").astype(str)
    if texts.str.len().sum() > 20:
        try:
            vec = TfidfVectorizer(stop_words="english", max_features=20)
            X = vec.fit_transform(texts)
            words = (
                pd.DataFrame({"term": vec.get_feature_names_out(), "importance": X.sum(axis=0).A1})
                .sort_values("importance", ascending=False)
            )
            fig = px.bar(words.head(15), x="term", y="importance", title="Top TF-IDF Feedback Terms")
            st.plotly_chart(fig, use_container_width=True)
        except ValueError:
            st.info("Feedback text is available, but not enough meaningful vocabulary was found for theme extraction.")
    else:
        st.info("Not enough feedback text for theme extraction.")
with tab2:
    st.subheader("Experience Trend")

    # Robust monthly trend logic.
    # Avoids pandas resample frequency issues across Streamlit Cloud / pandas versions.
    trend_df = df.copy()
    trend_df["feedback_date"] = pd.to_datetime(trend_df["feedback_date"], errors="coerce")
    trend_df = trend_df.dropna(subset=["feedback_date"])

    if trend_df.empty:
        st.info("No valid feedback dates available for trend analysis.")
    else:
        trend_df["month"] = trend_df["feedback_date"].dt.to_period("M").dt.to_timestamp()
        trend = (
            trend_df.groupby("month")
            .agg(
                records=("rating", "count"),
                experience=("experience_score", "mean"),
                avg_rating=("rating", "mean"),
                median_response_hours=("response_time_hours", "median"),
            )
            .reset_index()
            .sort_values("month")
        )

        fig = px.line(
            trend,
            x="month",
            y="experience",
            markers=True,
            title="Monthly Experience Score"
        )
        fig.update_layout(xaxis_title="Month", yaxis_title="Experience Score")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(trend.round(2), use_container_width=True, hide_index=True)
with tab3:
    st.subheader("Improvement Actions")
    for i,r in enumerate(recommendations(df),1):
        st.write(f"**{i}.** {r}")
    st.markdown("#### Sample Records")
    st.dataframe(df.head(80), use_container_width=True, hide_index=True)
