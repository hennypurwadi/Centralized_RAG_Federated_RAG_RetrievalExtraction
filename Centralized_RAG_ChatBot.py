
import streamlit as st
import os
import string
import time

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader

# Set page configuration
st.set_page_config(
    page_title="🐱 Centralized RAG Chatbot", #Click Win + . to show icons
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
    .source-header {
        font-size: 1.2rem;
        font-weight: bold;
        color: #1E88E5;
        margin-top: 1rem;
    }
    .source-text {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
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
</style>
""", unsafe_allow_html=True)

# Set OpenAI API key from Streamlit secrets or environment variable
def get_api_key():
    try:
        if 'OPENAI_API_KEY' in st.secrets:
            return st.secrets["OPENAI_API_KEY"]
    except:
        pass

    if 'OPENAI_API_KEY' in os.environ:
        return os.environ["OPENAI_API_KEY"]
    else:
        return st.sidebar.text_input("Enter your OpenAI API key:", type="password")

api_key = get_api_key()
if api_key:
    os.environ["OPENAI_API_KEY"] = api_key
else:
    st.warning("Please enter your OpenAI API key to continue.")
    st.stop()

# Initialize session state for storing index and chat history
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.vector_store = None
    st.session_state.qa_chain = None
    st.session_state.chat_history = []

# Load the text data from the file
@st.cache_data(show_spinner=False)
def load_data():
    text_loader = TextLoader(file_path="./data/msc_ai_hullonline_short.txt")
    return text_loader.load()

# Main function to build the Streamlit app
def main():
    # Header
    st.markdown("<h1 class='main-header'>🐱 Centralized RAG Chatbot</h1>", unsafe_allow_html=True)
    st.markdown("<h2 class='sub-header'>AI-Powered Question Answering System</h2>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: #666;'>MSc AI Online Program - University of Hull</h3>", unsafe_allow_html=True)

    # Sidebar for configuration
    st.sidebar.title("⚙️ Configuration")

    # Model settings (compact)
    with st.sidebar.expander("🤖 Model Settings", expanded=False):
        embeddings_model = st.selectbox(
            "Embeddings Model",
            ["text-embedding-3-small", "text-embedding-3-large"],
            index=0,
            help="Choose the OpenAI embedding model"
        )

        llm_model = st.selectbox(
            "LLM Model",
            ["gpt-4o", "gpt-3.5-turbo"],
            index=0,
            help="Choose the language model for generating responses"
        )

        temperature = st.slider(
            "Temperature",
            0.0, 1.0, 0.0, 0.1,
            help="Controls randomness in responses"
        )

    # Retrieval settings (compact)
    with st.sidebar.expander("🔍 Retrieval Settings", expanded=False):
        k_docs = st.slider(
            "Documents to retrieve",
            3, 15, 5,
            help="How many relevant document chunks to use"
        )

    # Document processing settings (compact)
    with st.sidebar.expander("📄 Document Processing", expanded=False):
        chunk_size = st.slider(
            "Chunk Size",
            300, 1000, 500, 50,
            help="Size of text chunks for processing"
        )
        chunk_overlap = st.slider(
            "Chunk Overlap",
            50, 200, 100, 10,
            help="Overlap between consecutive chunks"
        )

        data_file_path = st.text_input(
            "Data File Path",
            value="./data/msc_ai_hullonline_short.txt",
            help="Path to the text data file"
        )

    # Initialize button at the top
    if not st.session_state.initialized:
        if st.sidebar.button("🚀 Initialize RAG System", type="primary", use_container_width=True):
            with st.spinner("Initializing RAG System..."):
                # Initialize embedding model
                embeddings = OpenAIEmbeddings(model=embeddings_model)

                # Progress bar for initialization
                progress_bar = st.progress(0)

                # Load documents
                st.sidebar.write("Loading documents...")
                progress_bar.progress(20)
                documents = load_data()

                # Split documents
                st.sidebar.write("Processing documents...")
                progress_bar.progress(40)
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    separators=["\n\n", "\n", ".", "!", "?", ";", ":", " ", ""],
                    length_function=len
                )
                split_docs = text_splitter.split_documents(documents)
                st.sidebar.write(f"Created {len(split_docs)} document chunks")
                progress_bar.progress(60)

                # Build FAISS index
                st.sidebar.write("Building FAISS index...")
                vector_store = FAISS.from_documents(split_docs, embeddings)
                progress_bar.progress(80)

                # Create retriever and QA chain
                retriever = vector_store.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": k_docs}
                )

                qa_chain = RetrievalQA.from_chain_type(
                    llm=ChatOpenAI(model_name=llm_model, temperature=temperature),
                    chain_type="stuff",
                    retriever=retriever,
                    return_source_documents=True,
                    verbose=False
                )

                # Store in session state
                st.session_state.vector_store = vector_store
                st.session_state.qa_chain = qa_chain
                st.session_state.initialized = True
                st.session_state.split_docs_count = len(split_docs)

                progress_bar.progress(100)
                st.sidebar.success("✅ Initialization complete!")
                st.rerun()

        st.sidebar.info("👆 Click Initialize to start!")
    else:
        st.sidebar.success("✅ System Ready")
        if st.sidebar.button("🔄 Reset System", use_container_width=True):
            st.session_state.initialized = False
            st.session_state.vector_store = None
            st.session_state.qa_chain = None
            st.session_state.chat_history = []
            st.rerun()

    # Main chat interface
    if st.session_state.initialized:
        # Information box
        st.markdown("""
        <div class='info-box'>
            <strong>ℹ️ How it works:</strong> This RAG (Retrieval-Augmented Generation) system combines document retrieval
            with AI generation to provide accurate, contextual answers about the MSc AI Online program at the University of Hull.
        </div>
        """, unsafe_allow_html=True)

        # Display chat history
        if st.session_state.chat_history:
            st.subheader("💬 Chat History")
            for i, (question, answer, sources) in enumerate(st.session_state.chat_history):
                with st.container():
                    st.markdown(f"""
                    <div class='chat-container'>
                        <div class='question-text'>Q{i+1}: {question}</div>
                        <div class='answer-text'><strong>AI:</strong> {answer}</div>
                    </div>
                    """, unsafe_allow_html=True)

                    with st.expander(f"📚 View Sources for Q{i+1} ({len(sources)} documents)"):
                        for j, source in enumerate(sources, 1):
                            st.markdown(f"<div class='source-header'>Source {j}: {source.metadata.get('source', 'Unknown')}</div>", unsafe_allow_html=True)
                            st.markdown(f"<div class='source-text'>{source.page_content[:500]}...</div>", unsafe_allow_html=True)

                    st.markdown("---")

        # Question input section
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
            ask_button = st.button("Ask", type="primary", use_container_width=True)

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
            total_questions = len(st.session_state.chat_history)
            total_sources = sum(len(sources) for _, _, sources in st.session_state.chat_history)
            avg_sources = total_sources / total_questions if total_questions > 0 else 0

            col_stat1, col_stat2, col_stat3 = st.columns(3)
            with col_stat1:
                st.metric("Total Questions", total_questions)
            with col_stat2:
                st.metric("Total Sources Used", total_sources)
            with col_stat3:
                st.metric("Avg Sources per Question", f"{avg_sources:.1f}")

        # Use example question if set
        if hasattr(st.session_state, 'example_question'):
            user_question = st.session_state.example_question
            del st.session_state.example_question
            ask_button = True

        # Process question
        if ask_button and user_question:
            with st.spinner('🔍 Searching for relevant information and generating response...'):
                # Initialize progress bar
                progress_bar = st.progress(0)

                # Simulate progress for the query process
                for percent_complete in range(1, 51):
                    time.sleep(0.02)  # Simulate retrieval delay
                    progress_bar.progress(percent_complete)

                # Get response from QA chain
                response = st.session_state.qa_chain({"query": user_question})
                answer = response["result"]
                sources = response["source_documents"]

                # Continue progress simulation
                for percent_complete in range(51, 101):
                    time.sleep(0.01)  # Simulate processing delay
                    progress_bar.progress(percent_complete)

                # Add to chat history
                st.session_state.chat_history.append((user_question, answer, sources))

                # Show success message
                st.success(f"Generated response using {len(sources)} relevant sources!")

                # Clear input box, rerun to show the updated chat
                st.rerun()

        elif ask_button and not user_question:
            st.warning("⚠️ Please enter a question before asking!")

    else:
        # System not ready. Show initialization prompt
        st.markdown("""
        <div class='info-box'>
            <strong>🚀 Getting Started:</strong><br>
            1. Configure your settings in the sidebar<br>
            2. Enter your OpenAI API key if required<br>
            3. Click "Initialize RAG System" to load the data<br>
            4. Start asking questions about the MSc AI program!
        </div>
        """, unsafe_allow_html=True)

        # Show sample questions
        st.subheader("💡 Sample Questions You Could Ask:")
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
