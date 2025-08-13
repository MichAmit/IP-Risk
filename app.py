import streamlit as st
import pandas as pd
import requests
import spacy
from transformers import pipeline
from datetime import datetime, timedelta
import json

# --- CONFIGURATION ---
KEYWORDS = ["infringement", "injunction", "uspto", "opposition", "invalidated", "lawsuit", "litigation"]
RISK_WEIGHTS = {'sentiment': 0.6, 'keywords': 0.4}

# --- LOAD MODELS (cached for performance) ---
@st.cache_resource
def load_spacy_model():
    return spacy.load("en_core_web_sm")

@st.cache_resource
def load_sentiment_model():
    return pipeline("text-classification", model="yiyanghkust/finbert-tone", return_all_scores=True)

nlp = load_spacy_model()
sentiment_analyzer = load_sentiment_model()

# --- DATA FETCHING AND PROCESSING FUNCTIONS ---
# Replace your old fetch_gdelt_data function with this improved one

@st.cache_data(ttl=3600)
def fetch_gdelt_data(days=7):
    """Fetches business news from the last N days from GDELT."""
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days)

    start_str = start_date.strftime("%Y%m%d%H%M%S")
    end_str = end_date.strftime("%Y%m%d%H%M%S")

    query_keywords = " OR ".join([f'"{k}"' for k in KEYWORDS])
    query = f'({query_keywords}) sourcelang:english (theme:"BUSINESS" OR theme:"TECH_INTERNET")'

    url = (f"https://api.gdeltproject.org/api/v2/doc/doc?query={query}"
           f"&mode=ArtList&format=json&startdatetime={start_str}&enddatetime={end_str}&maxrecords=150")

    try:
        response = requests.get(url)
        response.raise_for_status() # Check for HTTP errors like 4xx/5xx
        data = response.json()      # Try to parse the JSON
        return data.get('articles', [])

    except requests.exceptions.HTTPError as e:
        st.error(f"🚨 An HTTP error occurred: {e}. The API might be temporarily down.")
        return []
    except requests.exceptions.JSONDecodeError:
        # This specifically catches the "empty box" error
        st.error("🚨 Failed to decode JSON. The GDELT API returned an empty or invalid response. Please try again later.")
        return []
    except requests.exceptions.RequestException as e:
        # This catches other network errors like timeouts
        st.error(f"🚨 A network connection error occurred: {e}")
        return []

def analyze_article(article_text):
    doc = nlp(article_text)
    companies = list(set([ent.text for ent in doc.ents if ent.label_ == "ORG"]))
    risky_sentences = [sent.text for sent in doc.sents if any(keyword in sent.text.lower() for keyword in KEYWORDS)]
    if not risky_sentences:
        return None, None, 0
    full_risk_text = " ".join(risky_sentences)
    sentiment_result = sentiment_analyzer(full_risk_text)[0]
    negative_score = next((item['score'] for item in sentiment_result if item['label'] == 'Negative'), 0)
    normalized_keyword_score = min(len(risky_sentences) / 5.0, 1.0)
    risk_score = (RISK_WEIGHTS['sentiment'] * negative_score) + (RISK_WEIGHTS['keywords'] * normalized_keyword_score)
    return companies, full_risk_text, risk_score

# --- STREAMLIT APP LAYOUT ---
st.set_page_config(page_title="IP-Risk Sentiment Analyzer", layout="wide")
st.title("⚖️ IP-Risk Sentiment Analyzer (Colab Edition)")
st.markdown("This tool scans recent news for IP risks and calculates a potential risk score for associated companies.")

if 'results_df' not in st.session_state:
    st.session_state.results_df = pd.DataFrame()

col1, col2 = st.columns([1, 4])
with col1:
    days_to_scan = st.slider("Days of news to scan:", 1, 30, 7)
    if st.button("Analyze News", type="primary"):
        with st.spinner("Analyzing articles..."):
            articles = fetch_gdelt_data(days=days_to_scan)
            results = [res for article in articles if (content := article.get('seendate', '') + " " + article.get('title', '') + " " + article.get('socialimage', '')) and len(content) > 100 and (analysis_result := analyze_article(content))[2] > 0.1 for company in analysis_result[0] if len(company) > 3 and " " in company and "LLC" not in company for res in [{'Company': company, 'Risk Score': analysis_result[2], 'Key Mentions': analysis_result[1], 'Source': article['url'], 'Date': article['seendate'][:8]}]]
            if results:
                df = pd.DataFrame(results)
                st.session_state.results_df = df.loc[df.groupby('Company')['Risk Score'].idxmax()].sort_values(by="Risk Score", ascending=False).reset_index(drop=True)
            else:
                st.session_state.results_df = pd.DataFrame()

with col2:
    st.header("Analysis Results")
    if not st.session_state.results_df.empty:
        st.dataframe(st.session_state.results_df, column_config={"Company": "🏢 Company", "Risk Score": st.column_config.ProgressColumn("📈 Risk Score", format="%.2f", min_value=0, max_value=1), "Key Mentions": "💬 Key Mentions", "Source": st.column_config.LinkColumn("🔗 Source", display_as="Link"), "Date": st.column_config.DateColumn("🗓️ Date", format="YYYY-MM-DD")}, use_container_width=True, hide_index=True)
    else:
        st.info("Click 'Analyze News' to fetch and display IP risk data.")
