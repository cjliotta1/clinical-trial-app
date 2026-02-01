import gradio as gr
import pandas as pd
import requests
import matplotlib.pyplot as plt
from google import genai  # The universal way
import os

# This looks for the "Secret" key you added to Streamlit settings
try:
    api_key = os.environ.get("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)
except:
    client = None

def run_clinical_analysis(target_indication):
    if not client:
        return "Error: API Key not found in Streamlit Secrets!", None, None, None
    
    # --- AI OVERVIEW ---
    response = client.models.generate_content(
        model="gemini-2.5-flash", 
        contents=f"Provide a clean summary of {target_indication}. Use bolding, but no hashtags."
    )
    disease_overview = response.text

    # --- FETCH TRIALS (ClinicalTrials.gov) ---
    url = "https://clinicaltrials.gov/api/v2/studies"
    params = {'query.cond': target_indication, 'pageSize': 15, 'format': 'json'}
    
    try:
        r = requests.get(url, params=params)
        trials = r.json().get('studies', [])
        data = [{
            'Drug': t['protocolSection'].get('armsInterventionsModule', {}).get('interventions', [{}])[0].get('name', 'N/A'),
            'Sponsor': t['protocolSection']['sponsorCollaboratorsModule']['leadSponsor']['name'],
            'Phase': t['protocolSection']['designModule'].get('phases', ['N/A'])[0]
        } for t in trials]
        df = pd.DataFrame(data).drop_duplicates()
    except:
        df = pd.DataFrame([{"Error": "Database timeout"}])

    # --- MOA ANALYSIS ---
    moa_response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=f"Summarize the Mechanisms of Action for {target_indication} therapies."
    )
    moa_analysis = moa_response.text

    # --- CHART ---
    fig, ax = plt.subplots(figsize=(5, 3))
    if not df.empty and 'Phase' in df.columns:
        df['Phase'].value_counts().plot(kind='pie', ax=ax, autopct='%1.1f%%')
    ax.set_ylabel('')
    
    return disease_overview, df, moa_analysis, fig

# --- INTERFACE ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🏥 Clinical Analysis Hub")
    with gr.Row():
        search_input = gr.Textbox(label="Disease Name")
        submit_btn = gr.Button("Analyze", variant="primary")
    with gr.Row():
        out_text = gr.Markdown()
        out_plot = gr.Plot()
    with gr.Row():
        out_table = gr.Dataframe()
        out_ai = gr.Markdown()

    submit_btn.click(fn=run_clinical_analysis, inputs=search_input, outputs=[out_text, out_table, out_ai, out_plot])

demo.launch()
