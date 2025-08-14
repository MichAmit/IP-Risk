# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
from bs4 import BeautifulSoup
import requests, datetime, re, time
import numpy as np

# ---------- CONFIG ----------
st.set_page_config(page_title="IP-Risk Pulse", layout="wide")
st.title("🔍  IP-Risk Sentiment Monitor")
st.caption("Live patent-litigation news feed (last 30 days)")

# ---------- DATA FETCH ----------
@st.cache_data(ttl=3600)   # cache for 1 hour
def fetch_and_score():
    KEYWORDS = ["patent infringement","patent opposition","lawsuit","USPTO","WIPO","EPO"]
    START = (datetime.datetime.utcnow() - datetime.timedelta(days=30)).strftime("%Y-%m-%d")

    # Google News RSS – no API key needed
    import feedparser
    q = " OR ".join([f'"{k}"' for k in KEYWORDS])
    url = f"https://news.google.com/rss/search?q={requests.utils.quote(q)}&hl=en-US&gl=US&ceid=US:en"
    feed = feedparser.parse(url)

    risk_kw = ["infringement","lawsuit","sued","opposition","injunction","invalidated"]
    def score(title):
        t = title.lower()
        return min(1.0, sum(k in t for k in risk_kw) / 6.0)

    rows = []
    for entry in feed.entries[:50]:
        rows.append({
            "title": entry.title,
            "url":   entry.link,
            "date":  pd.to_datetime(entry.published),
            "risk":  score(entry.title)
        })
        time.sleep(0.1)  # polite crawl
    return pd.DataFrame(rows)

df = fetch_and_score()

# ---------- SIDEBAR ----------
st.sidebar.metric("Articles analysed", len(df))
threshold = st.sidebar.slider("Risk threshold", 0.0, 1.0, 0.0)

# ---------- CHART ----------
fig = px.scatter(df[df["risk"] >= threshold],
                 x="date", y="risk",
                 hover_data=["title"],
                 color="risk",
                 color_continuous_scale="Reds",
                 title="Risk over time")
fig.update_layout(xaxis_title="Date", yaxis_title="Risk Score")
st.plotly_chart(fig, use_container_width=True)

# ---------- TABLE ----------
st.subheader("Top risky articles")
top = df.sort_values("risk", ascending=False).head(10)[["title","risk","url"]]
st.dataframe(top, use_container_width=True)

# ---------- DOWNLOAD ----------
csv = df.to_csv(index=False)
st.download_button("Download CSV", csv, "ip_risk_report.csv", mime="text/csv")
