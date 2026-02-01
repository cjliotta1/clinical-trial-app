import gradio as gr
import pandas as pd
import requests
import matplotlib.pyplot as plt
from google.colab import ai 

def run_clinical_analysis(target_indication):
    if not target_indication:
        return "Please enter a disease.", None, "No data.", None

    # AI OVERVIEW (Formatted for Markdown)
    disease_overview = ai.generate_text(
        f"Provide a clean, professional summary of {target_indication}. Use bolding for key terms, but no raw hashtags.", 
        model_name='google/gemini-2.5-flash'
    )

    # FETCH TRIALS (Same logic as before)
    clintrial_base_url = "https://clinicaltrials.gov/api/v2/studies"
    params = {'query.cond': target_indication, 'pageSize': 15, 'format': 'json'}
    
    try:
        response = requests.get(clintrial_base_url, params=params)
        trials = response.json().get('studies', [])
        df = pd.DataFrame([
            {
                'Drug': t['protocolSection'].get('armsInterventionsModule', {}).get('interventions', [{}])[0].get('name', 'N/A'),
                'Sponsor': t['protocolSection']['sponsorCollaboratorsModule']['leadSponsor']['name'],
                'Phase': t['protocolSection']['designModule'].get('phases', ['N/A'])[0]
            } for t in trials
        ]).drop_duplicates()
    except:
        df = pd.DataFrame([{"Error": "Could not connect to database"}])

    # MOA ANALYSIS
    moa_analysis = ai.generate_text(f"Summarize the MOA for {target_indication} in a clear list.")

    # PIE CHART
    fig, ax = plt.subplots(figsize=(5, 3))
    if not df.empty and 'Phase' in df.columns:
        df['Phase'].value_counts().plot(kind='pie', ax=ax, autopct='%1.1f%%', colors=['#ff9999','#66b3ff','#99ff99'])
    ax.set_ylabel('')
    
    return disease_overview, df, moa_analysis, fig

# --- THE NEW LEGIBLE INTERFACE ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🏥 Universal Clinical Insights Tool")
    
    with gr.Row():
        search_input = gr.Textbox(label="Enter any Disease Name", placeholder="e.g. Lupus, Alzheimer's...")
        submit_btn = gr.Button("Search & Analyze", variant="primary")
    
    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("### 📝 Disease Overview")
            out_text = gr.Markdown() # Legible formatting here
        with gr.Column(scale=1):
            gr.Markdown("### 📊 Phase Distribution")
            out_plot = gr.Plot()

    with gr.Row():
        with gr.Column():
            gr.Markdown("### 💊 Industry Trial Pipeline")
            out_table = gr.Dataframe()
        with gr.Column():
            gr.Markdown("### 🔬 MOA Summary")
            out_ai = gr.Markdown() # Legible formatting here

    submit_btn.click(fn=run_clinical_analysis, inputs=search_input, outputs=[out_text, out_table, out_ai, out_plot])

demo.launch(share=True)
