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

# Custom CSS for compact layout
st.markdown("""
<style>
    .stRadio > div {
        flex-direction: row;
        gap: 10px;
    }
    .stRadio > div > label {
        margin-right: 15px;
    }
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    .element-container {
        margin-bottom: 0.5rem;
    }
    h3 {
        margin-bottom: 0.5rem;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Informed Consent
st.markdown("""
### Informed Consent

Thank you for your interest in this study.

This research is conducted by MSc AI Online students at the University of Hull to evaluate the performance of different chatbot systems. Your participation is voluntary, and you may withdraw at any time before submitting the survey. You may also skip any question you do not wish to answer. All responses are anonymous and will only be used for academic purposes.

Please confirm that you have read and understood the above information and that you agree to participate in this survey.
""")
consent_given = st.checkbox("I agree and give my informed consent to participate in this survey.")

if not consent_given:
    st.warning("You must provide consent to continue.")
    st.stop()

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

# Rating options
rating_options = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]

# Create three columns for compact layout
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### Centralized RAG")
    
    relevance_centralized = st.radio(
        "Relevance (0-10)", 
        rating_options, 
        index=rating_options.index(str(st.session_state.survey_data.get("relevance_centralized", 5))),
        key="relevance_centralized",
        horizontal=True
    )
    st.session_state.survey_data["relevance_centralized"] = int(relevance_centralized)
    
    coherence_centralized = st.radio(
        "Coherence (0-10)", 
        rating_options, 
        index=rating_options.index(str(st.session_state.survey_data.get("coherence_centralized", 5))),
        key="coherence_centralized",
        horizontal=True
    )
    st.session_state.survey_data["coherence_centralized"] = int(coherence_centralized)
    
    ambiguity_centralized = st.radio(
        "Ambiguity Handling (0-10)", 
        rating_options, 
        index=rating_options.index(str(st.session_state.survey_data.get("ambiguity_centralized", 5))),
        key="ambiguity_centralized",
        horizontal=True
    )
    st.session_state.survey_data["ambiguity_centralized"] = int(ambiguity_centralized)
    
    satisfaction_centralized = st.radio(
        "User Satisfaction (0-10)", 
        rating_options, 
        index=rating_options.index(str(st.session_state.survey_data.get("satisfaction_centralized", 5))),
        key="satisfaction_centralized",
        horizontal=True
    )
    st.session_state.survey_data["satisfaction_centralized"] = int(satisfaction_centralized)

with col2:
    st.markdown("### Federated RAG")
    
    relevance_federated = st.radio(
        "Relevance (0-10)", 
        rating_options, 
        index=rating_options.index(str(st.session_state.survey_data.get("relevance_federated", 5))),
        key="relevance_federated",
        horizontal=True
    )
    st.session_state.survey_data["relevance_federated"] = int(relevance_federated)
    
    coherence_federated = st.radio(
        "Coherence (0-10)", 
        rating_options, 
        index=rating_options.index(str(st.session_state.survey_data.get("coherence_federated", 5))),
        key="coherence_federated",
        horizontal=True
    )
    st.session_state.survey_data["coherence_federated"] = int(coherence_federated)
    
    ambiguity_federated = st.radio(
        "Ambiguity Handling (0-10)", 
        rating_options, 
        index=rating_options.index(str(st.session_state.survey_data.get("ambiguity_federated", 5))),
        key="ambiguity_federated",
        horizontal=True
    )
    st.session_state.survey_data["ambiguity_federated"] = int(ambiguity_federated)
    
    satisfaction_federated = st.radio(
        "User Satisfaction (0-10)", 
        rating_options, 
        index=rating_options.index(str(st.session_state.survey_data.get("satisfaction_federated", 5))),
        key="satisfaction_federated",
        horizontal=True
    )
    st.session_state.survey_data["satisfaction_federated"] = int(satisfaction_federated)

with col3:
    st.markdown("### Retrieval-Based")
    
    relevance_retrieval = st.radio(
        "Relevance (0-10)", 
        rating_options, 
        index=rating_options.index(str(st.session_state.survey_data.get("relevance_retrieval", 5))),
        key="relevance_retrieval",
        horizontal=True
    )
    st.session_state.survey_data["relevance_retrieval"] = int(relevance_retrieval)
    
    coherence_retrieval = st.radio(
        "Coherence (0-10)", 
        rating_options, 
        index=rating_options.index(str(st.session_state.survey_data.get("coherence_retrieval", 5))),
        key="coherence_retrieval",
        horizontal=True
    )
    st.session_state.survey_data["coherence_retrieval"] = int(coherence_retrieval)
    
    ambiguity_retrieval = st.radio(
        "Ambiguity Handling (0-10)", 
        rating_options, 
        index=rating_options.index(str(st.session_state.survey_data.get("ambiguity_retrieval", 5))),
        key="ambiguity_retrieval",
        horizontal=True
    )
    st.session_state.survey_data["ambiguity_retrieval"] = int(ambiguity_retrieval)
    
    satisfaction_retrieval = st.radio(
        "User Satisfaction (0-10)", 
        rating_options, 
        index=rating_options.index(str(st.session_state.survey_data.get("satisfaction_retrieval", 5))),
        key="satisfaction_retrieval",
        horizontal=True
    )
    st.session_state.survey_data["satisfaction_retrieval"] = int(satisfaction_retrieval)

# Feedback section
st.markdown("### Open Feedback")
feedback = st.text_area("Please share any feedback:", height=80, key="open_feedback")
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
if st.button("Submit Survey", type="primary"):
    required_scores = [f"{c}_{b}" for c in ["relevance", "coherence", "ambiguity", "satisfaction"] for b in ["centralized", "federated", "retrieval"]]
    if all(k in st.session_state.survey_data for k in required_scores):
        count = save_survey_data(st.session_state.survey_data)
        st.success(f"Submitted! Response #{count}")
        st.download_button(
            label="Download Your Response",
            data=json.dumps(st.session_state.survey_data, indent=2),
            file_name=f"survey_response_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
    else:
        st.error("Please complete all ratings before submitting.")

