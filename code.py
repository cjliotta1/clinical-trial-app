import streamlit as st
import pandas as pd
import requests
from google import genai

st.set_page_config(page_title="Clinical Trial AI", layout="wide")

# 1. API Key Check
api_key = st.secrets.get("GEMINI_API_KEY")
if not api_key:
    st.error("Missing API Key in Streamlit Secrets!")
    st.stop()

client = genai.Client(api_key=api_key)

# 2. UI Header
st.title("🔬 Clinical Trial Analysis Dashboard")
indication = st.text_input("Disease Name", placeholder="e.g. Psoriasis")

if st.button("Run Analysis"):
    if indication:
        col1, col2 = st.columns([2, 1])
        
        with st.spinner("Analyzing..."):
            # --- AI SECTION ---
            res = client.models.generate_content(
                model="gemini-2.0-flash", 
                contents=f"Summary of {indication}"
            )
            with col1:
                st.subheader("📖 Overview")
                st.markdown(res.text)

            # --- DATA SECTION ---
            url = "https://clinicaltrials.gov/api/v2/studies"
            params = {'query.cond': indication, 'pageSize': 10, 'format': 'json'}
            r = requests.get(url, params=params)
            trials = r.json().get('studies', [])
            
            trial_list = []
            for t in trials:
                ps = t.get('protocolSection', {})
                trial_list.append({
                    'Drug': ps.get('armsInterventionsModule', {}).get('interventions', [{}])[0].get('name', 'N/A'),
