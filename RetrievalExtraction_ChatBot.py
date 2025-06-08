import streamlit as st
import os
import pickle
from sentence_transformers import SentenceTransformer, util
import time

# Set page configuration
st.set_page_config(
    page_title="🐱 Retrieval-Based Chatbot",
    page_icon="./icon.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS for enhanced styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-container {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 1rem;
        margin: 1rem 0;
        border-left: 4px solid #1E88E5;
    }
    .question-text {
        font-weight: bold;
        color: #1565C0;
        margin-bottom: 0.5rem;
    }
    .answer-text {
        color: #2E7D32;
        line-height: 1.6;
        background-color: #E8F5E8;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-top: 0.5rem;
    }
    .similarity-score {
        background-color: #E3F2FD;
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin-top: 0.5rem;
        font-size: 0.9rem;
        color: #1565C0;
    }
    .info-box {
        background-color: #FFF3E0;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #FF9800;
        margin: 1rem 0;
    }
    .stats-container {
        background-color: #F3E5F5;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .button-container {
        display: flex;
        gap: 10px;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Function to load the saved embeddings
@st.cache_data(show_spinner=False)
def load_embeddings(embedding_file_path):
    try:
        with open(embedding_file_path, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error(f"Embedding file not found: {embedding_file_path}")
        return None

# Function to load the ground truth answers from a file
@st.cache_data(show_spinner=False)
def load_ground_truth_answers(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            answers = file.readlines()
        return [answer.strip() for answer in answers if answer.strip()]
    except FileNotFoundError:
        st.error(f"Ground truth file not found: {file_path}")
        return []

# Function to load the embedding model
@st.cache_resource(show_spinner=False)
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

# Function to generate embeddings and find the most relevant answers
def perform_embeddings(question_embedding, answer_embeddings, top_k=3):
    # Compute cosine similarity between the question and answers
    similarities = util.pytorch_cos_sim(question_embedding, answer_embeddings)
    
    # Get top-k most similar answers
    top_results = similarities.topk(k=top_k)
    
    return top_results

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'system_ready' not in st.session_state:
    st.session_state.system_ready = False

def main():
    # Header
    st.markdown("<h1 class='main-header'>🐱 Retrieval-Based Chatbot</h1>", unsafe_allow_html=True)
    st.markdown("<h2 class='sub-header'>Embeddings-Based Question Answering System</h2>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: #666;'>MSc AI Online Program - University of Hull</h3>", unsafe_allow_html=True)
    
    # Sidebar configuration
    st.sidebar.title("⚙️ Configuration")
    
    # Model settings
    st.sidebar.subheader("Model Settings")
    embedding_model = st.sidebar.selectbox(
        "Embedding Model",
        ["all-MiniLM-L6-v2", "all-mpnet-base-v2", "all-distilroberta-v1"],
        index=0,
        help="Choose the sentence transformer model for embeddings"
    )
    
    # Retrieval settings
    st.sidebar.subheader("Retrieval Settings")
    top_k = st.sidebar.slider(
        "Number of top answers to retrieve",
        min_value=1,
        max_value=10,
        value=3,
        help="How many most similar answers to show"
    )
    
    similarity_threshold = st.sidebar.slider(
        "Similarity Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.05,
        help="Minimum similarity score to consider an answer relevant"
    )
    
    # File paths
    st.sidebar.subheader("Data Files")
    embedding_file_path = st.sidebar.text_input(
        "Embedding File Path",
        value="./data/MiniLM_embeddings.pkl",
        help="Path to the pre-computed embeddings file"
    )
    
    ground_truth_file_path = st.sidebar.text_input(
        "Ground Truth File Path",
        value="./data/msc_ai_hullonline_short.txt",
        help="Path to the ground truth answers file"
    )
    
    # System initialization
    if not st.session_state.system_ready:
        st.sidebar.subheader("System Status")
        if st.sidebar.button("🚀 Initialize System", type="primary"):
            with st.spinner("Initializing retrieval system..."):
                progress_bar = st.progress(0)
                
                # Load embeddings
                st.sidebar.write("Loading embeddings...")
                progress_bar.progress(25)
                answer_embeddings = load_embeddings(embedding_file_path)
                
                # Load ground truth answers
                st.sidebar.write("Loading ground truth answers...")
                progress_bar.progress(50)
                ground_truth_answers = load_ground_truth_answers(ground_truth_file_path)
                
                # Load model
                st.sidebar.write("Loading embedding model...")
                progress_bar.progress(75)
                model = load_model()
                
                progress_bar.progress(100)
                
                if answer_embeddings is not None and ground_truth_answers:
                    st.session_state.answer_embeddings = answer_embeddings
                    st.session_state.ground_truth_answers = ground_truth_answers
                    st.session_state.model = model
                    st.session_state.system_ready = True
                    st.sidebar.success("✅ System initialized successfully!")
                    st.rerun()
                else:
                    st.sidebar.error("❌ Failed to initialize system. Check file paths.")
    else:
        st.sidebar.success("✅ System Ready")
        
        # System statistics
        st.sidebar.subheader("📊 System Statistics")
        st.sidebar.metric("Total Answers", len(st.session_state.ground_truth_answers))
        st.sidebar.metric("Embedding Dimensions", st.session_state.answer_embeddings.shape[1])
        st.sidebar.metric("Questions Asked", len(st.session_state.chat_history))
        
        # Reset button
        if st.sidebar.button("🔄 Reset System"):
            st.session_state.system_ready = False
            st.session_state.chat_history = []
            if 'answer_embeddings' in st.session_state:
                del st.session_state.answer_embeddings
            if 'ground_truth_answers' in st.session_state:
                del st.session_state.ground_truth_answers
            if 'model' in st.session_state:
                del st.session_state.model
            st.rerun()
    
    # Main interface
    if st.session_state.system_ready:
        # Information box
        st.markdown("""
        <div class='info-box'>
            <strong>ℹ️ How it works:</strong> This system uses pre-computed embeddings to find the most semantically similar answers 
            to your questions about the MSc AI Online program at the University of Hull. Ask any question below!
        </div>
        """, unsafe_allow_html=True)
        
        # Display chat history
        if st.session_state.chat_history:
            st.subheader("💬 Chat History")
            for i, (question, answers, scores) in enumerate(st.session_state.chat_history):
                with st.container():
                    st.markdown(f"""
                    <div class='chat-container'>
                        <div class='question-text'>Q{i+1}: {question}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    for j, (answer, score) in enumerate(zip(answers, scores)):
                        st.markdown(f"""
                        <div class='answer-text'>
                            <strong>Answer {j+1}:</strong> {answer}
                        </div>
                        <div class='similarity-score'>
                            📊 Similarity Score: {score:.4f}
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("---")
        
        # Question input
        st.subheader("🤔 Ask a Question")
        
        # Create columns for better layout
        col1, col2 = st.columns([4, 1])
        
        with col1:
            user_question = st.text_input(
                "Enter your question about the MSc AI program:",
                placeholder="e.g., How long does the MSc Artificial Intelligence course take to complete?",
                key="question_input"
            )
        
        with col2:
            st.write("")  # Add some spacing
            ask_button = st.button("🔍 Ask", type="primary", use_container_width=True)
        
        # Additional action buttons
        col3, col4, col5 = st.columns(3)
        with col3:
            clear_history = st.button("🗑️ Clear History", use_container_width=True)
        with col4:
            example_question = st.button("💡 Example Question", use_container_width=True)
        with col5:
            show_stats = st.button("📈 Show Statistics", use_container_width=True)
        
        # Handle button actions
        if example_question:
            st.session_state.example_question = "How long does the MSc Artificial Intelligence course take to complete?"
            st.rerun()
        
        if clear_history:
            st.session_state.chat_history = []
            st.rerun()
        
        if show_stats and st.session_state.chat_history:
            st.subheader("📊 Session Statistics")
            avg_similarity = sum([max(scores) for _, _, scores in st.session_state.chat_history]) / len(st.session_state.chat_history)
            st.metric("Average Best Similarity Score", f"{avg_similarity:.4f}")
            st.metric("Total Questions", len(st.session_state.chat_history))
        
        # Use example question if set
        if hasattr(st.session_state, 'example_question'):
            user_question = st.session_state.example_question
            del st.session_state.example_question
            ask_button = True
        
        # Process question
        if ask_button and user_question:
            with st.spinner('🔍 Searching for the best answers...'):
                # Progress bar
                progress_bar = st.progress(0)
                
                # Generate embedding for the user's question
                progress_bar.progress(30)
                question_embedding = st.session_state.model.encode([user_question], convert_to_tensor=True)
                
                # Find the most relevant answers
                progress_bar.progress(60)
                top_results = perform_embeddings(
                    question_embedding, 
                    st.session_state.answer_embeddings, 
                    top_k=top_k
                )
                
                progress_bar.progress(90)
                
                # Extract results
                scores = top_results.values[0].cpu().numpy()
                indices = top_results.indices[0].cpu().numpy()
                
                # Filter by similarity threshold
                valid_results = [(idx, score) for idx, score in zip(indices, scores) if score >= similarity_threshold]
                
                progress_bar.progress(100)
                
                if valid_results:
                    answers = [st.session_state.ground_truth_answers[idx] for idx, _ in valid_results]
                    filtered_scores = [score for _, score in valid_results]
                    
                    # Add to chat history
                    st.session_state.chat_history.append((user_question, answers, filtered_scores))
                    
                    # Display immediate results
                    st.success(f"✅ Found {len(valid_results)} relevant answer(s)!")
                    
                else:
                    st.warning(f"⚠️ No answers found above similarity threshold ({similarity_threshold:.2f})")
                
                # Clear input and refresh
                st.rerun()
        
        elif ask_button and not user_question:
            st.warning("⚠️ Please enter a question before asking!")
    
    else:
        # System not ready - show initialization prompt
        st.markdown("""
        <div class='info-box'>
            <strong>🚀 Getting Started:</strong><br>
            1. Check the file paths in the sidebar<br>
            2. Click "Initialize System" to load the embeddings and data<br>
            3. Start asking questions about the MSc AI program!
        </div>
        """, unsafe_allow_html=True)
        
        # Show sample questions
        st.subheader("💡 Sample Questions You Can Ask:")
        sample_questions = [
            "How long does the MSc Artificial Intelligence course take to complete?",
            "What is the total cost of the MSc Artificial Intelligence Online program?",
            "What is the minimum academic qualification required to apply for the MSc Artificial Intelligence (Online) program?",
            "What if I don't have a degree but have relevant professional experience?",
            "What language proficiency is required if my first language isn't English?"
        ]
        
        for i, question in enumerate(sample_questions, 1):
            st.write(f"{i}. {question}")

if __name__ == "__main__":
    main()

