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

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

def llm(prompt, model="gpt-4o-mini"):
    res = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return res.choices[0].message.content

# -----------------------
# Sidebar
# -----------------------
st.sidebar.header("Controls")
target_indication = st.sidebar.text_input(
    "Target indication", "hidradenitis suppurativa"
)
max_trials = st.sidebar.slider("Number of recent trials", 5, 50, 20)
run = st.sidebar.button("Run analysis")

# -----------------------
# Main
# -----------------------
if run:
    # Disease overview
    st.subheader("Disease overview")
    with st.spinner("Thinking..."):
        disease_overview = llm(
            f"Describe {target_indication}. Symptoms, prognosis, and standard of care."
        )
    st.markdown(disease_overview)

    # Fetch trials
    st.subheader("Clinical trials")
    base_url = "https://clinicaltrials.gov/api/v2/studies"
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
    r = requests.get(base_url, params=params)
    trials = r.json().get("studies", [])
    trials = sorted(
        trials,
        key=lambda x: x["protocolSection"]["statusModule"]
        .get("startDateStruct", {})
        .get("date", ""),
        reverse=True,
    )[:max_trials]
    st.write(f"Found {len(trials)} trials")

    # Extract interventions
    interventions = []
    for t in trials:
        try:
            p = t["protocolSection"]
            arms = p.get("armsInterventionsModule", {}).get("armGroups", [])
            exp_arms = [
                a for a in arms if a.get("type") in ["EXPERIMENTAL", "ACTIVE_COMPARATOR"]
            ]
            exp_labels = [a["label"] for a in exp_arms]
            ints = p.get("armsInterventionsModule", {}).get("interventions", [])
            for i in ints:
                if i.get("type") == "DRUG":
                    if any(lbl in exp_labels for lbl in i.get("armGroupLabels", [])):
                        interventions.append(
                            {
                                "drug": i.get("name", "Unknown"),
                                "description": i.get("description", ""),
                                "phase": p["designModule"].get("phases", ["Unknown"])[0],
                                "year": p["statusModule"]
                                .get("startDateStruct", {})
                                .get("date", "")[:4],
                                "sponsor": p["sponsorCollaboratorsModule"]["leadSponsor"]["name"],
                            }
                        )
        except:
            pass

    df = (
        pd.DataFrame(interventions)
        .drop_duplicates(subset=["drug", "sponsor", "phase"])
        .reset_index(drop=True)
    )
    st.dataframe(df)

    # MOA analysis
    summary = "\n".join(
        f"- {r.drug} ({r.sponsor}, Phase {r.phase}, {r.year})"
        for _, r in df.head(15).iterrows()
    )

    st.subheader("MOA landscape")
    with st.spinner("Analyzing MOAs..."):
        moa_analysis = llm(
            f"""
Disease: {target_indication}

Biology:
{disease_overview}

Drug candidates:
{summary}

Analyze:
1. Mechanisms being pursued
2. First-in-class candidates
3. Unexplored mechanisms
4. Most differentiated approach
"""
        )
    st.markdown(moa_analysis)

    # MOA classification
    with st.spinner("Classifying drugs..."):
        moa_labels = llm(
            f"""
Assign ONE primary MOA to each drug:

{summary}

Format:
Drug: MOA
"""
        )

    moa_map = {}
    for line in moa_labels.splitlines():
        if ":" in line:
            d, m = line.split(":", 1)
            moa_map[d.strip()] = m.strip()

    df["moa"] = df["drug"].map(moa_map).fillna("Other")
    st.dataframe(df[["drug", "phase", "moa"]])

    # Chart
    st.subheader("MOA by phase")
    phase_moa = df.groupby(["phase", "moa"]).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(10, 6))
    phase_moa.plot(kind="bar", stacked=True, ax=ax)
    ax.set_ylabel("Number of candidates")
    ax.set_title("Mechanism of Action by Phase")
    plt.xticks(rotation=0)
    st.pyplot(fig)
