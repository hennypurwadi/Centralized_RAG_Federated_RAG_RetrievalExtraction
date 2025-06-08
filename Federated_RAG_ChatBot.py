
import streamlit as st
import os
import sys
from bs4 import BeautifulSoup
import string
import requests
from urllib.parse import urljoin, urlparse
import time  # For simulating progress

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS  # Fixed import from community package
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
from langchain.retrievers import MergerRetriever

# Set page configuration
st.set_page_config(
    page_title="🐱 Federated RAG Chatbot", #Click Win + . to show icons
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
    .university-badge {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-size: 0.8rem;
        font-weight: bold;
        margin-right: 0.5rem;
    }
    .hull-badge {
        background-color: #E3F2FD;
        color: #1565C0;
    }
    .keele-badge {
        background-color: #F3E5F5;
        color: #7B1FA2;
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

# Helper functions for web scraping and document processing
def clean_html(raw_html: str) -> str:
    """
    Remove boilerplate HTML tags (script, style, nav, footer, header) and return clean text.
    """
    soup = BeautifulSoup(raw_html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)

def chunk_documents(docs, chunk_size=1100, chunk_overlap=200):
    """Split each LangChain Document into smaller chunks for embeddings."""
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = []
    for doc in docs:
        # Split page_content into pieces
        pieces = splitter.split_text(doc.page_content)
        for piece in pieces:
            chunks.append(Document(page_content=piece, metadata=doc.metadata))
    return chunks

def scrape_website(url, max_depth=1):
    """Custom web scraper that supports recursive crawling with depth control."""
    visited = set()
    documents = []
    domain = urlparse(url).netloc

    def scrape(current_url, depth):
        if depth > max_depth or current_url in visited:
            return

        visited.add(current_url)
        st.sidebar.write(f"Scraping: {current_url}")

        try:
            # Use a realistic user agent
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
            response = requests.get(current_url, timeout=10, headers=headers)
            response.encoding = response.apparent_encoding  # Fix encoding issues
            response.raise_for_status()

            content = clean_html(response.text)  # Clean HTML

            # Create document with metadata
            doc = Document(
                page_content=content,
                metadata={"source": current_url, "title": current_url.split("/")[-1]}
            )
            documents.append(doc)

            # Only follow links if not at max depth
            if depth < max_depth:
                soup = BeautifulSoup(response.text, "html.parser")
                for link in soup.find_all("a", href=True):
                    href = link["href"]
                    full_url = urljoin(current_url, href)

                    # Only follow links within the same domain
                    if urlparse(full_url).netloc == domain and full_url not in visited:
                        scrape(full_url, depth + 1)

        except Exception as e:
            st.sidebar.write(f"Error scraping {current_url}: {str(e)}")

    # Start scraping from the initial URL
    scrape(url, 0)
    return documents

def get_federated_docs(urls, max_depth=1):
    """
    Crawl a list of URLs up to max_depth and return a list of LangChain Document objects.
    """
    all_docs = []
    for url in urls:
        docs = scrape_website(url, max_depth=max_depth)
        all_docs.extend(docs)
    return all_docs

# Initialize session state for storing indices
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.hull_index = None
    st.session_state.keele_index = None
    st.session_state.combined_retriever = None
    st.session_state.qa_chain = None
    st.session_state.chat_history = []

# Main function to build the Streamlit app
def main():
    # Header
    st.markdown("<h1 class='main-header'>🐱 Federated RAG Chatbot</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: #666;'>MSc Artificial Intelligence online at University of Hull &</h3>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: #666;'>MSc Computer Science with Artificial Intelligence online at Keele University</h3>", unsafe_allow_html=True)

    # Sidebar for configuration
    st.sidebar.title("⚙️ Configuration")

    # Model settings (compact)
    with st.sidebar.expander("🐱 Model Settings", expanded=False):
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
            0.0, 1.0, 0.3, 0.1,
            help="Controls randomness in responses"
        )

    # Retrieval settings (compact)
    with st.sidebar.expander("🔍 Retrieval Settings", expanded=False):
        k_docs = st.slider(
            "Documents per source",
            3, 15, 5,
            help="How many relevant document chunks to retrieve from each university"
        )

    # University data sources (compact)
    with st.sidebar.expander("🏫 Data Sources", expanded=False):
        st.write("**University of Hull:**")
        st.write("• MSc Artificial Intelligence")
        st.write("• Course information & FAQs")
        st.write("• Funding & costs")

        st.write("**Keele University:**")
        st.write("• MSc Computer Science with AI")
        st.write("• Online study information")
        st.write("• Program details")

    # Define URLs for Hull and Keele
    hull_urls = [
        "https://online.hull.ac.uk",
        "https://online.hull.ac.uk/contact",
        "https://online.hull.ac.uk/why-join-us/faqs",
        "https://online.hull.ac.uk/funding-options",
        "https://online.hull.ac.uk/course-costs",
        "https://online.hull.ac.uk/courses/msc-artificial-intelligence"
    ]

    keele_urls = [
        "https://online.keele.ac.uk/online-programme/msc-computer-science-with-artificial-intelligence",
        "https://online.keele.ac.uk/online-study",
        "https://online.keele.ac.uk/about-us"
    ]

    # Initialize button at the top
    if not st.session_state.initialized:
        if st.sidebar.button("🚀 Initialize Federated RAG System", type="primary", use_container_width=True):
            with st.spinner("Initializing Federated RAG System..."):
                # Initialize embedding model
                embeddings = OpenAIEmbeddings(model=embeddings_model)

                # Progress bar for initialization
                progress_bar = st.progress(0)

                # Load, clean, and chunk documents for Hull
                st.sidebar.write("Loading Hull documents...")
                progress_bar.progress(10)
                hull_raw = get_federated_docs(hull_urls, max_depth=1)
                progress_bar.progress(30)
                hull_chunks = chunk_documents(hull_raw)
                st.sidebar.write(f"Created {len(hull_chunks)} Hull chunks.")
                progress_bar.progress(40)

                # Load, clean, and chunk documents for Keele
                st.sidebar.write("Loading Keele documents...")
                progress_bar.progress(50)
                keele_raw = get_federated_docs(keele_urls, max_depth=1)
                progress_bar.progress(70)
                keele_chunks = chunk_documents(keele_raw)
                st.sidebar.write(f"Created {len(keele_chunks)} Keele chunks.")
                progress_bar.progress(80)

                # Build FAISS indexes
                st.sidebar.write("Building FAISS indices...")
                hull_index = FAISS.from_documents(hull_chunks, embeddings)
                keele_index = FAISS.from_documents(keele_chunks, embeddings)
                progress_bar.progress(90)

                # Build retrievers
                hull_retriever = hull_index.as_retriever(search_kwargs={"k": k_docs})
                keele_retriever = keele_index.as_retriever(search_kwargs={"k": k_docs})

                # Composite retriever
                combined_retriever = MergerRetriever(retrievers=[hull_retriever, keele_retriever])

                # Build RetrievalQA chain
                qa_chain = RetrievalQA.from_chain_type(
                    llm=ChatOpenAI(model_name=llm_model, temperature=temperature),
                    retriever=combined_retriever,
                    return_source_documents=True,
                    chain_type="stuff",
                )

                # Store in session state
                st.session_state.hull_index = hull_index
                st.session_state.keele_index = keele_index
                st.session_state.combined_retriever = combined_retriever
                st.session_state.qa_chain = qa_chain
                st.session_state.initialized = True
                st.session_state.hull_chunks_count = len(hull_chunks)
                st.session_state.keele_chunks_count = len(keele_chunks)

                progress_bar.progress(100)
                st.sidebar.success("Initialization complete!")
                st.rerun()

        st.sidebar.info("👆 Click Initialize to start!")
    else:
        st.sidebar.success("✅ System Ready")
        if st.sidebar.button("🔄 Reset System", use_container_width=True):
            st.session_state.initialized = False
            st.session_state.hull_index = None
            st.session_state.keele_index = None
            st.session_state.combined_retriever = None
            st.session_state.qa_chain = None
            st.session_state.chat_history = []
            st.rerun()

    # System statistics (if initialized)
    if st.session_state.initialized:
        st.sidebar.divider()
        st.sidebar.subheader("📊 System Stats")
        col1, col2 = st.sidebar.columns(2)
        with col1:
            st.metric("Hull Chunks", st.session_state.get('hull_chunks_count', 'N/A'))
            st.metric("Questions", len(st.session_state.chat_history))
        with col2:
            st.metric("Keele Chunks", st.session_state.get('keele_chunks_count', 'N/A'))
            st.metric("LLM", llm_model.split('-')[0])

    # Main chat interface
    if st.session_state.initialized:
        # Information box
        st.markdown("""
        <div class='info-box'>
            <strong>ℹ️ How it works:</strong> This Federated RAG system combines information from multiple universities
            to provide comprehensive comparisons between Hull's MSc AI and Keele's MSc Computer Science with AI programs.
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
                        hull_sources = []
                        keele_sources = []

                        for source in sources:
                            if 'hull.ac.uk' in source.metadata.get('source', ''):
                                hull_sources.append(source)
                            elif 'keele.ac.uk' in source.metadata.get('source', ''):
                                keele_sources.append(source)

                        if hull_sources:
                            st.markdown("**🎓 University of Hull Sources:**")
                            for j, source in enumerate(hull_sources, 1):
                                st.markdown(f"<span class='university-badge hull-badge'>Hull {j}</span>", unsafe_allow_html=True)
                                st.markdown(f"**Source:** {source.metadata.get('source', 'Unknown')}")
                                st.markdown(f"<div class='source-text'>{source.page_content[:400]}...</div>", unsafe_allow_html=True)

                        if keele_sources:
                            st.markdown("**🎓 Keele University Sources:**")
                            for j, source in enumerate(keele_sources, 1):
                                st.markdown(f"<span class='university-badge keele-badge'>Keele {j}</span>", unsafe_allow_html=True)
                                st.markdown(f"**Source:** {source.metadata.get('source', 'Unknown')}")
                                st.markdown(f"<div class='source-text'>{source.page_content[:400]}...</div>", unsafe_allow_html=True)

                    st.markdown("---")

        # Question input section
        st.subheader("🤔 Ask a Question")

        # Create columns for better layout
        col1, col2 = st.columns([4, 1])

        with col1:
            user_question = st.text_input(
                "Compare programs or ask about Hull/Keele universities:",
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
            total_questions = len(st.session_state.chat_history)
            total_sources = sum(len(sources) for _, _, sources in st.session_state.chat_history)
            avg_sources = total_sources / total_questions if total_questions > 0 else 0

            # Count sources by university
            hull_source_count = 0
            keele_source_count = 0
            for _, _, sources in st.session_state.chat_history:
                for source in sources:
                    if 'hull.ac.uk' in source.metadata.get('source', ''):
                        hull_source_count += 1
                    elif 'keele.ac.uk' in source.metadata.get('source', ''):
                        keele_source_count += 1

            col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
            with col_stat1:
                st.metric("Total Questions", total_questions)
            with col_stat2:
                st.metric("Hull Sources Used", hull_source_count)
            with col_stat3:
                st.metric("Keele Sources Used", keele_source_count)
            with col_stat4:
                st.metric("Avg Sources/Question", f"{avg_sources:.1f}")

        # Use example question if set
        if hasattr(st.session_state, 'example_question'):
            user_question = st.session_state.example_question
            del st.session_state.example_question
            ask_button = True

        # Process question
        if ask_button and user_question:
            with st.spinner('🔍 Searching across universities and generating response...'):
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

                # Show success message with source breakdown
                hull_count = sum(1 for s in sources if 'hull.ac.uk' in s.metadata.get('source', ''))
                keele_count = sum(1 for s in sources if 'keele.ac.uk' in s.metadata.get('source', ''))
                st.success(f"✅ Generated response using {len(sources)} sources ({hull_count} Hull, {keele_count} Keele)!")

                # Clear the input box and rerun to show the updated chat
                st.rerun()

        elif ask_button and not user_question:
            st.warning("⚠️ Please enter a question before asking!")

    else:
        # System not ready - show initialization prompt
        st.markdown("""
        <div class='info-box'>
            <strong>🚀 Getting Started:</strong><br>
            1. Configure your settings in the sidebar<br>
            2. Enter your OpenAI API key if required<br>
            3. Click "Initialize Federated RAG System" to load data from both universities<br>
            4. Start comparing Hull and Keele AI programs!
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
