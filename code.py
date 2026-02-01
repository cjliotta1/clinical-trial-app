# app.py

import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
from openai import OpenAI

# -----------------------
# App setup
# -----------------------
st.set_page_config(page_title="Clinical Trial MOA Mapper", layout="wide")
st.title("🧬 AI-Powered Clinical Trial MOA Mapping")

# OpenAI client (expects OPENAI_API_KEY in Streamlit secrets)
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# -----------------------
# LLM helper (ROBUST)
# -----------------------
def llm(prompt, model="gpt-4.1-mini"):
    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.3,
    )

    # Safely extract text from response
    output_text = ""
    for item in response.output:
        if item["type"] == "output_text":
            output_text += item["text"]

    return output_text.strip()

# -----------------------
# Sidebar
# -----------------------
st.sidebar.header("Controls")
target_indication = st.sidebar.text_input(
    "Target indication (any disease)",
    "hidradenitis suppurativa"
)
run = st.sidebar.button("Run analysis")

# -----------------------
# Main workflow
# -----------------------
if run:

    # -----------------------
    # Disease overview
    # -----------------------
    st.subheader("1️⃣ Disease overview")
    with st.spinner("Thinking..."):
        disease_overview = llm(
            f"Describe {target_indication}. Symptoms, prognosis, and standard of care."
        )
    st.markdown(disease_overview)

    # -----------------------
    # Fetch ALL trials (pagination)
    # -----------------------
    st.subheader("2️⃣ Fetching clinical trials")

    base_url = "https://clinicaltrials.gov/api/v2/studies"
    all_trials = []
    page_token = None

    with st.spinner("Downloading trials from clinicaltrials.gov..."):
        while True:
            params = {
                "query.cond": target_indication,
                "filter.advanced": (
                    "AREA[LeadSponsorClass]INDUSTRY "
                    "AND AREA[StartDate]RANGE[2018-01-01,MAX] "
                    "AND (AREA[Phase]PHASE2 OR AREA[Phase]PHASE3)"
                ),
                "pageSize": 100,
                "format": "json",
            }

            if page_token:
                params["pageToken"] = page_token

            r = requests.get(base_url, params=params)
            data = r.json()

            all_trials.extend(data.get("studies", []))
            page_token = data.get("nextPageToken")

            if not page_token:
                break

    st.success(f"Found {len(all_trials)} Phase 2/3 industry-sponsored trials")

    # -----------------------
    # Extract interventions
    # -----------------------
    st.subheader("3️⃣ Extracting drug candidates")

    interventions = []

    for trial in all_trials:
        try:
            protocol = trial["protocolSection"]

            arms = protocol.get("armsInterventionsModule", {}).get("armGroups", [])
            experimental_arms = [
                a for a in arms if a.get("type") in ["EXPERIMENTAL", "ACTIVE_COMPARATOR"]
            ]
            experimental_labels = [a["label"] for a in experimental_arms]

            trial_interventions = protocol.get(
                "armsInterventionsModule", {}
            ).get("interventions", [])

            for intervention in trial_interventions:
                if intervention.get("type") == "DRUG":
                    arm_labels = intervention.get("armGroupLabels", [])
                    if any(lbl in experimental_labels for lbl in arm_labels):
                        interventions.append({
                            "drug": intervention.get("name", "Unknown"),
                            "description": intervention.get("description", ""),
                            "phase": protocol["designModule"].get("phases", ["Unknown"])[0],
                            "year": protocol["statusModule"]
                            .get("startDateStruct", {})
                            .get("date", "")[:4],
                            "sponsor": protocol["sponsorCollaboratorsModule"]
                            ["leadSponsor"]["name"],
                        })
        except Exception:
            continue

    df = (
        pd.DataFrame(interventions)
        .drop_duplicates(subset=["drug", "sponsor", "phase"])
        .reset_index(drop=True)
    )

    st.write(f"Extracted {len(df)} unique experimental drug candidates")
    st.dataframe(df)

    # -----------------------
    # MOA analysis (summary)
    # -----------------------
    st.subheader("4️⃣ MOA landscape analysis")

    intervention_summary = "\n".join(
        f"- {row.drug} ({row.sponsor}, Phase {row.phase}, {row.year})"
        for _, row in df.head(20).iterrows()
    )

    with st.spinner("Analyzing mechanisms of action..."):
        moa_analysis = llm(
            f"""
Disease: {target_indication}

Biology:
{disease_overview}

Drug candidates:
{intervention_summary}

Analyze:
1. Mechanisms being pursued
2. First-in-class drugs
3. Unexplored mechanisms
4. Most differentiated approach
"""
        )

    st.markdown(moa_analysis)

    # -----------------------
    # MOA classification
    # -----------------------
    st.subheader("5️⃣ MOA classification")

    with st.spinner("Classifying drugs..."):
        moa_labels = llm(
            f"""
For each drug below, assign ONE primary mechanism of action category.

{intervention_summary}

Format:
DrugName: MOA
"""
        )

    moa_map = {}
    for line in moa_labels.splitlines():
        if ":" in line:
            drug, moa = line.split(":", 1)
            moa_map[drug.strip()] = moa.strip()

    df["moa"] = df["drug"].map(moa_map).fillna("Other")
    st.dataframe(df[["drug", "phase", "moa"]])

    # -----------------------
    # Chart
    # -----------------------
    st.subheader("6️⃣ MOA by development phase")

    phase_moa = df.groupby(["phase", "moa"]).size().unstack(fill_value=0)

    fig, ax = plt.subplots(figsize=(10, 6))
    phase_moa.plot(kind="bar", stacked=True, ax=ax)
    ax.set_title("Mechanism of Action by Phase")
    ax.set_xlabel("Phase")
    ax.set_ylabel("Number of candidates")
    plt.xticks(rotation=0)
    plt.tight_layout()

    st.pyplot(fig)

    st.success("Analysis complete 🎉")
