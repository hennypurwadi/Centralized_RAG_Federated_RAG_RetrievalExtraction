
import streamlit as st
import pandas as pd
import datetime
import json
import os

st.set_page_config(
    page_title="Chatbot Evaluation Survey",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Header
st.markdown("""
<div style="text-align: center; margin-bottom: 1rem;">
    <h3>Chatbot Evaluation Survey</h3>
    <p><em>MSc AI Online – University of Hull</em></p>
</div>
""", unsafe_allow_html=True)

# Session init
if 'survey_data' not in st.session_state:
    st.session_state.survey_data = {}

# Chatbot rating section
st.markdown("### Centralized RAG")

relevance_centralized = st.slider("Relevance (0-10)", 0, 10, st.session_state.survey_data.get("relevance_centralized", 5), key="relevance_centralized")
st.session_state.survey_data["relevance_centralized"] = relevance_centralized

coherence_centralized = st.slider("Coherence (0-10)", 0, 10, st.session_state.survey_data.get("coherence_centralized", 5), key="coherence_centralized")
st.session_state.survey_data["coherence_centralized"] = coherence_centralized

ambiguity_centralized = st.slider("Ambiguity Handling (0-10)", 0, 10, st.session_state.survey_data.get("ambiguity_centralized", 5), key="ambiguity_centralized")
st.session_state.survey_data["ambiguity_centralized"] = ambiguity_centralized

satisfaction_centralized = st.slider("User Satisfaction (0-10)", 0, 10, st.session_state.survey_data.get("satisfaction_centralized", 5), key="satisfaction_centralized")
st.session_state.survey_data["satisfaction_centralized"] = satisfaction_centralized
st.markdown("### Federated RAG")

relevance_federated = st.slider("Relevance (0-10)", 0, 10, st.session_state.survey_data.get("relevance_federated", 5), key="relevance_federated")
st.session_state.survey_data["relevance_federated"] = relevance_federated

coherence_federated = st.slider("Coherence (0-10)", 0, 10, st.session_state.survey_data.get("coherence_federated", 5), key="coherence_federated")
st.session_state.survey_data["coherence_federated"] = coherence_federated

ambiguity_federated = st.slider("Ambiguity Handling (0-10)", 0, 10, st.session_state.survey_data.get("ambiguity_federated", 5), key="ambiguity_federated")
st.session_state.survey_data["ambiguity_federated"] = ambiguity_federated

satisfaction_federated = st.slider("User Satisfaction (0-10)", 0, 10, st.session_state.survey_data.get("satisfaction_federated", 5), key="satisfaction_federated")
st.session_state.survey_data["satisfaction_federated"] = satisfaction_federated
st.markdown("### Retrieval-Based")

relevance_retrieval = st.slider("Relevance (0-10)", 0, 10, st.session_state.survey_data.get("relevance_retrieval", 5), key="relevance_retrieval")
st.session_state.survey_data["relevance_retrieval"] = relevance_retrieval

coherence_retrieval = st.slider("Coherence (0-10)", 0, 10, st.session_state.survey_data.get("coherence_retrieval", 5), key="coherence_retrieval")
st.session_state.survey_data["coherence_retrieval"] = coherence_retrieval

ambiguity_retrieval = st.slider("Ambiguity Handling (0-10)", 0, 10, st.session_state.survey_data.get("ambiguity_retrieval", 5), key="ambiguity_retrieval")
st.session_state.survey_data["ambiguity_retrieval"] = ambiguity_retrieval

satisfaction_retrieval = st.slider("User Satisfaction (0-10)", 0, 10, st.session_state.survey_data.get("satisfaction_retrieval", 5), key="satisfaction_retrieval")
st.session_state.survey_data["satisfaction_retrieval"] = satisfaction_retrieval


# Feedback section
st.markdown("### Open Feedback")
feedback = st.text_area("Please share any feedback:", height=100, key="open_feedback")
st.session_state.survey_data['open_feedback'] = feedback

# Function to save data
def save_survey_data(data):
    data['timestamp'] = datetime.datetime.now().isoformat()
    df = pd.DataFrame([data])
    csv_file = "survey_responses.csv"
    if os.path.exists(csv_file):
        existing_df = pd.read_csv(csv_file)
        df = pd.concat([existing_df, df], ignore_index=True)
    df.to_csv(csv_file, index=False)
    return len(df)

# Submit button
if st.button("Submit Survey"):
    required_scores = [f"{c}_{b}" for c in ["relevance", "coherence", "ambiguity", "satisfaction"] for b in ["centralized", "federated", "retrieval"]]
    if all(k in st.session_state.survey_data for k in required_scores):
        count = save_survey_data(st.session_state.survey_data)
        st.success(f"Submitted! Response #{count}")
        st.download_button(
            label="📥 Download Your Response",
            data=json.dumps(st.session_state.survey_data, indent=2),
            file_name=f"survey_response_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
    else:
        st.error("Please complete all ratings before submitting.")
