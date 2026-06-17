import pandas as pd
import numpy as np

AREAS=["Learner Support","Assessment Helpdesk","Technical Support","Content Access","Certification","General Service"]
POS=["helpful","clear","quick","resolved","smooth","professional","easy","supportive"]
NEG=["delay","confusing","unresolved","slow","unclear","frustrating","difficult","repeated","poor"]

def generate_synthetic_experience_data(n=2500, seed=22):
    rng=np.random.default_rng(seed)
    dates=pd.Timestamp.today().normalize()-pd.to_timedelta(rng.integers(0,365,n), unit="D")
    area=rng.choice(AREAS,n)
    rating=np.clip(np.round(rng.normal(3.7,1.1,n)),1,5).astype(int)
    texts=[]
    for r in rating:
        words=rng.choice(POS if r>=4 else NEG if r<=2 else POS+NEG, size=4, replace=True)
        texts.append("Service experience was " + " ".join(words))
    return pd.DataFrame({
        "feedback_date":dates,"service_area":area,"feedback_text":texts,"rating":rating,
        "channel":rng.choice(["Email","Portal","Chat","Survey","Call"],n),
        "status":rng.choice(["Resolved","Open","Escalated","In Review"],n,p=[.62,.18,.08,.12]),
        "response_time_hours":np.clip(rng.gamma(2.0,8,n)+(6-rating)*2,1,120).round(1)
    })

def prepare_experience_data(df):
    df=df.copy(); cols={c.lower().strip().replace(" ","_"):c for c in df.columns}
    def pick(ns):
        for n in ns:
            if n in cols: return cols[n]
        return None
    n=len(df); out=pd.DataFrame()
    out["feedback_date"]=pd.to_datetime(df[pick(["feedback_date","date","created_date","submitted_date"])], errors="coerce") if pick(["feedback_date","date","created_date","submitted_date"]) else pd.Timestamp.today()
    out["service_area"]=df[pick(["service_area","category","department","service"])].astype(str) if pick(["service_area","category","department","service"]) else "General Service"
    out["feedback_text"]=df[pick(["feedback_text","comment","text","description","feedback"])].astype(str) if pick(["feedback_text","comment","text","description","feedback"]) else ""
    out["rating"]=pd.to_numeric(df[pick(["rating","score","satisfaction"])], errors="coerce") if pick(["rating","score","satisfaction"]) else 3
    out["channel"]=df[pick(["channel","source"])].astype(str) if pick(["channel","source"]) else "Survey"
    out["status"]=df[pick(["status","state"])].astype(str) if pick(["status","state"]) else "Resolved"
    out["response_time_hours"]=pd.to_numeric(df[pick(["response_time_hours","response_time","duration_hours"])], errors="coerce") if pick(["response_time_hours","response_time","duration_hours"]) else 24
    out["feedback_date"]=out["feedback_date"].fillna(pd.Timestamp.today())
    out["rating"]=out["rating"].fillna(3).clip(1,5)
    out["response_time_hours"]=out["response_time_hours"].fillna(24).clip(0,999)
    return out

def add_sentiment(df):
    df=df.copy()
    text=df["feedback_text"].str.lower()
    pos=sum(text.str.contains(w, regex=False).astype(int) for w in POS)
    neg=sum(text.str.contains(w, regex=False).astype(int) for w in NEG)
    df["sentiment_score"]=(pos-neg).astype(float)
    df["experience_score"]=((df["rating"]/5)*55 + (df["sentiment_score"]+4).clip(0,8)/8*25 + (1-df["response_time_hours"].rank(pct=True))*20).round(1)
    df["risk_band"]=pd.cut(df["experience_score"], bins=[-1,45,70,101], labels=["High Risk","Watch","Healthy"])
    return df

def date_filter(df, mode):
    if mode=="All Data": return df
    days={"Last 30 Days":30,"Last 90 Days":90,"Last 180 Days":180}[mode]
    return df[df["feedback_date"] >= df["feedback_date"].max()-pd.Timedelta(days=days)]

def recommendations(df):
    rec=[]
    low=df[df["experience_score"]<45]
    if len(low)>0:
        area=low["service_area"].value_counts().index[0]
        rec.append(f"Prioritize **{area}**; it has the highest number of high-risk experience records.")
    if df["response_time_hours"].median()>24:
        rec.append("Median response time exceeds 24 hours. Review routing and response ownership.")
    if (df["rating"]<=2).mean()>0.2:
        rec.append("Low ratings exceed 20%. Add root-cause review for repeated negative feedback.")
    rec.append("Track experience score weekly and review service areas moving from Healthy to Watch.")
    return rec
