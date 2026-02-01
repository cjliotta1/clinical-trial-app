import streamlit as st
import pandas as pd
import requests
from google import genai
import os

# 1. Page Configuration
st.set_page_config(page_title="Clinical Trial AI", layout="wide")

# 2. Get API Key from Streamlit Secrets
api_key = st.secrets.get("GEMINI_API_KEY")

if not api_key:
    st.error("🚨 Missing API Key! Go to Streamlit Settings > Secrets and add: GEMINI_API_KEY = 'your_key_here'")
    st.stop()

# 3. Initialize Gemini
client = genai.Client(api_key=api_key)

# 4. The User Interface
st.title("🔬 Clinical Trial Analysis Dashboard")
st.write("Enter a disease name to get an AI summary and recent industry trials.")

# Search Box
indication = st.text_input("Disease Name", placeholder="e.g. Psoriasis, Lupus...")

if st.button("Run Analysis"):
    if not indication:
        st.warning("Please enter a disease name first.")
    else:
        # Create two columns for the layout
        col1, col2 = st.columns([2, 1])
        
        with st.spinner("AI is analyzing..."):
            try:
                # Part A: AI Overview
                response = client.models.generate_content(
                    model="gemini-2.0-flash", 
                    contents=f"Provide a clean, professional summary of {indication}. Use bolding for headers."
                )
                
                with col1:
                    st.subheader("📖 Disease Overview")
                    st.markdown(response.text)

                # Part B: Clinical Trials Data
                url = "https://clinicaltrials.gov/api/v2/studies"
                params = {'query.cond': indication, 'pageSize': 10, 'format': 'json'}
                r = requests.get(url, params=params)
                trials = r.json().get('studies', [])
                
                trial_data = []
                for t in trials:
                    protocol = t.get('protocolSection', {})
                    trial_data.append({
                        'Drug': protocol.get('armsInterventionsModule', {}).get('interventions', [{}])[0].get('name', 'N/A'),
                        'Sponsor': protocol.get('sponsorCollaboratorsModule', {}).get('leadSponsor', {}).get('name', 'N/A'),
                        'Phase': protocol.get('designModule', {}).get('phases', ['N/A'])[0]
                    })

                with col2:
                    st.subheader("📊 Industry Pipeline")
                    if trial_data:
                        df = pd.DataFrame(trial_data)
                        st.dataframe(df, use_container_width=True)
                        
                        # Simple Phase Chart
                        phase_counts = df['Phase'].value_counts()
                        st.bar_chart(phase_counts)
                    else:
                        st.info("No recent industry trials found for this search.")
