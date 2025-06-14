
import streamlit as st
import os
import sys
from bs4 import BeautifulSoup
import string
import requests
from urllib.parse import urljoin, urlparse
import time
import json
from typing import List, Dict, Any

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA

# Set page configuration
st.set_page_config(
    page_title=" 🐭True Federated RAG Chatbot",
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
    .federated-node {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 1rem;
        margin: 1rem 0;
        border: 3px solid #FF9800;
        border-left: 8px solid #FF9800;
    }
    .hull-node {
        border-color: #1E88E5;
        border-left-color: #1E88E5;
    }
    .keele-node {
        border-color: #7B1FA2;
        border-left-color: #7B1FA2;
    }
    .private-label {
        background-color: #FF5722;
        color: white;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-size: 0.8rem;
        font-weight: bold;
        margin-bottom: 1rem;
        display: inline-block;
    }
    .node-results {
        background-color: #E8F5E8;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-top: 1rem;
    }
    .aggregation-section {
        background-color: #E3F2FD;
        padding: 1.5rem;
        border-radius: 1rem;
        margin: 1rem 0;
        border-left: 8px solid #2196F3;
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
</style>
""", unsafe_allow_html=True)

# Set OpenAI API key
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

# Helper functions for document processing
def clean_html(raw_html: str) -> str:
    soup = BeautifulSoup(raw_html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)

def chunk_documents(docs, chunk_size=1100, chunk_overlap=200):
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = []
    for doc in docs:
        pieces = splitter.split_text(doc.page_content)
        for piece in pieces:
            chunks.append(Document(page_content=piece, metadata=doc.metadata))
    return chunks

def scrape_website(url, max_depth=1):
    visited = set()
    documents = []
    domain = urlparse(url).netloc

    def scrape(current_url, depth):
        if depth > max_depth or current_url in visited:
            return
        visited.add(current_url)
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            response = requests.get(current_url, timeout=10, headers=headers)
            response.encoding = response.apparent_encoding
            response.raise_for_status()
            content = clean_html(response.text)
            doc = Document(
                page_content=content,
                metadata={"source": current_url, "title": current_url.split("/")[-1]}
            )
            documents.append(doc)
            if depth < max_depth:
                soup = BeautifulSoup(response.text, "html.parser")
                for link in soup.find_all("a", href=True):
                    href = link["href"]
                    full_url = urljoin(current_url, href)
                    if urlparse(full_url).netloc == domain and full_url not in visited:
                        scrape(full_url, depth + 1)
        except Exception as e:
            st.sidebar.write(f"Error scraping {current_url}: {str(e)}")

    scrape(url, 0)
    return documents

def get_federated_docs(urls, max_depth=1):
    all_docs = []
    for url in urls:
        docs = scrape_website(url, max_depth=max_depth)
        all_docs.extend(docs)
    return all_docs

# TRUE FEDERATED RAG CLASSES
class FederatedNode:
    """Represents an independent federated node (university) with its own private data and retrieval system"""

    def __init__(self, name: str, urls: List[str], embeddings_model: str = "text-embedding-3-small"):
        self.name = name
        self.urls = urls
        self.embeddings = OpenAIEmbeddings(model=embeddings_model)
        self.vector_store = None
        self.retriever = None
        self.documents = []
        self.is_initialized = False

    def initialize_node(self, k_docs: int = 5):
        """Initialize the federated node with its own private data"""
        st.sidebar.write(f"Initializing {self.name} node (PRIVATE)...")

        # Load and process documents locally
        raw_docs = get_federated_docs(self.urls, max_depth=1)
        self.documents = chunk_documents(raw_docs)

        # Build local vector store (PRIVATE)
        self.vector_store = FAISS.from_documents(self.documents, self.embeddings)
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": k_docs})

        self.is_initialized = True
        st.sidebar.write(f"{self.name} node ready ({len(self.documents)} chunks)")

    def local_retrieve(self, query: str) -> List[Dict[str, Any]]:
        """Perform local retrieval and return ONLY the results (not raw data)"""
        if not self.is_initialized:
            raise ValueError(f"{self.name} node not initialized")

        # Perform local retrieval
        local_docs = self.retriever.get_relevant_documents(query)

        # Return only the results (no access to raw vector store)
        results = []
        for i, doc in enumerate(local_docs):
            results.append({
                "content": doc.page_content[:400] + "..." if len(doc.page_content) > 400 else doc.page_content,  # Limit content length
                "source": doc.metadata.get("source", "Unknown"),
                "node": self.name,
                "rank": i + 1
            })

        return results

class FederationCoordinator:
    """Coordinates federated queries across multiple independent nodes"""

    def __init__(self, nodes: List[FederatedNode], llm_model: str = "gpt-4o", temperature: float = 0.3):
        self.nodes = nodes
        self.llm = ChatOpenAI(model_name=llm_model, temperature=temperature)

    def federated_query(self, query: str, max_response_length: int = 200) -> Dict[str, Any]:
        """Send query to all federated nodes and aggregate results"""
        all_results = []
        node_responses = {}

        # Send query to each federated node independently
        for node in self.nodes:
            st.write(f"Querying {node.name} node...")
            node_results = node.local_retrieve(query)
            node_responses[node.name] = node_results
            all_results.extend(node_results)

        # Aggregate and rank results (this is where federation happens)
        aggregated_context = self._aggregate_results(all_results)

        # Generate response using aggregated context
        prompt = f"""Based on the federated search results from multiple universities, provide a CONCISE answer (max {max_response_length} words) to: {query}

Federated Results:
{aggregated_context}

Requirements:
- Keep response under {max_response_length} words
- Focus on key facts only
- Compare universities when relevant
- Be direct and specific
- Avoid lengthy explanations"""

        response = self.llm.predict(prompt)

        return {
            "query": query,
            "answer": response,
            "node_responses": node_responses,
            "total_sources": len(all_results)
        }

    def _aggregate_results(self, results: List[Dict[str, Any]]) -> str:
        """Aggregate results from all federated nodes"""
        aggregated = ""
        for i, result in enumerate(results, 1):
            aggregated += f"\n[Source {i} - {result['node']}]: {result['content'][:300]}...\n"
        return aggregated

# Initialize session state
if 'federated_initialized' not in st.session_state:
    st.session_state.federated_initialized = False
    st.session_state.hull_node = None
    st.session_state.keele_node = None
    st.session_state.coordinator = None
    st.session_state.chat_history = []

def main():
    # Header
    st.markdown("<h1 class='main-header'>True Federated RAG Chatbot</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: #666;'>MSc Artificial Intelligence online at University of Hull & </h3>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: #666;'>MSc Computer Science with Artificial Intelligence online at Keele University</h3>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: #666;'>Independent University Nodes with Complete Data Privacy</h3>", unsafe_allow_html=True)
    


    # Information about true federation
    st.markdown("""
    <div style='background-color: #FFF3E0; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #FF9800; margin: 1rem 0;'>
        <strong>True Federated Architecture:</strong> Each university maintains its own private data and retrieval system.
        Only final results are shared - no raw data or direct access to other nodes' vector stores.
    </div>
    """, unsafe_allow_html=True)

    # Sidebar configuration
    st.sidebar.title("Federated Configuration")

    with st.sidebar.expander("Model Settings", expanded=False):
        embeddings_model = st.selectbox(
            "Embeddings Model",
            ["text-embedding-3-small", "text-embedding-3-large"],
            index=0
        )
        llm_model = st.selectbox(
            "LLM Model",
            ["gpt-4o", "gpt-3.5-turbo"],
            index=0
        )
        temperature = st.slider("Temperature", 0.0, 1.0, 0.3, 0.1)

    with st.sidebar.expander("Retrieval Settings", expanded=False):
        k_docs = st.slider("Documents per node", 3, 10, 5)

    # Define federated nodes
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

    # Initialize federated system
    if not st.session_state.federated_initialized:
        if st.sidebar.button("🚀 Initialize Federated System", type="primary", use_container_width=True):
            with st.spinner("Initializing True Federated RAG System..."):
                progress_bar = st.progress(0)

                # Create independent federated nodes
                st.session_state.hull_node = FederatedNode("Hull University", hull_urls, embeddings_model)
                progress_bar.progress(20)

                st.session_state.keele_node = FederatedNode("Keele University", keele_urls, embeddings_model)
                progress_bar.progress(40)

                # Initialize each node independently
                st.session_state.hull_node.initialize_node(k_docs)
                progress_bar.progress(70)

                st.session_state.keele_node.initialize_node(k_docs)
                progress_bar.progress(90)

                # Create federation coordinator
                st.session_state.coordinator = FederationCoordinator(
                    [st.session_state.hull_node, st.session_state.keele_node],
                    llm_model,
                    temperature
                )
                progress_bar.progress(100)

                st.session_state.federated_initialized = True
                st.sidebar.success("Federated System Ready!")
                st.rerun()

        st.sidebar.info("👆 Click to initialize federated nodes")
    else:
        st.sidebar.success("Federated System Active")
        if st.sidebar.button("Reset System", use_container_width=True):
            st.session_state.federated_initialized = False
            st.session_state.hull_node = None
            st.session_state.keele_node = None
            st.session_state.coordinator = None
            st.session_state.chat_history = []
            st.rerun()

    # Show federated node status
    if st.session_state.federated_initialized:
        st.sidebar.divider()
        st.sidebar.subheader("Federated Nodes")

        # Hull node status
        st.sidebar.markdown("**Hull University Node**")
        st.sidebar.write(f"• Status: {'Active' if st.session_state.hull_node.is_initialized else '❌ Inactive'}")
        st.sidebar.write(f"• Documents: {len(st.session_state.hull_node.documents)}")
        st.sidebar.write(f"• Privacy: Complete")

        # Keele node status
        st.sidebar.markdown("**Keele University Node**")
        st.sidebar.write(f"• Status: {'Active' if st.session_state.keele_node.is_initialized else '❌ Inactive'}")
        st.sidebar.write(f"• Documents: {len(st.session_state.keele_node.documents)}")
        st.sidebar.write(f"• Privacy: Complete")

        st.sidebar.write(f"**Total Queries:** {len(st.session_state.chat_history)}")

    # Main interface
    if st.session_state.federated_initialized:
        # Show federated architecture
        st.subheader("Federated Node Architecture")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            <div class='federated-node hull-node'>
                <div class='private-label'>PRIVATE NODE</div>
                <h4>Hull University Node</h4>
                <p><strong>Local Data:</strong> MSc AI program information</p>
                <p><strong>Vector Store:</strong> Private FAISS index</p>
                <p><strong>Access:</strong> Results only, no raw data sharing</p>
                <div class='node-results'>
                    <strong>Node Status:</strong> Independent & Secure
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div class='federated-node keele-node'>
                <div class='private-label'>PRIVATE NODE</div>
                <h4>Keele University Node</h4>
                <p><strong>Local Data:</strong> MSc CS with AI program information</p>
                <p><strong>Vector Store:</strong> Private FAISS index</p>
                <p><strong>Access:</strong> Results only, no raw data sharing</p>
                <div class='node-results'>
                    <strong>Node Status:</strong> Independent & Secure
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Federation coordinator section
        st.markdown("""
        <div class='aggregation-section'>
            <h4>Federation Coordinator</h4>
            <p><strong>Function:</strong> Distributes queries to all nodes and aggregates ONLY the final results</p>
            <p><strong>Privacy:</strong> No access to raw data or vector stores from any node</p>
            <p><strong>Process:</strong> Query → Distribute → Local Retrieval → Result Aggregation → Generation</p>
        </div>
        """, unsafe_allow_html=True)

        # Chat interface
        st.subheader("💬 Federated Query Interface")

        # Display chat history
        if st.session_state.chat_history:
            for i, chat in enumerate(st.session_state.chat_history):
                with st.container():
                    st.markdown(f"""
                    <div class='chat-container'>
                        <div class='question-text'>Q{i+1}: {chat['query']}</div>
                        <div class='answer-text'><strong>Federated AI:</strong> {chat['answer']}</div>
                    </div>
                    """, unsafe_allow_html=True)

                    with st.expander(f"View Federated Results for Q{i+1}"):
                        for node_name, results in chat['node_responses'].items():
                            st.markdown(f"**🏫 {node_name} Results:**")
                            for j, result in enumerate(results, 1):
                                st.markdown(f"**Result {j}:** {result['content'][:200]}...")
                                st.markdown(f"*Source: {result['source']}*")
                            st.markdown("---")

        # Query input
        col1, col2 = st.columns([4, 1])

        with col1:
            user_question = st.text_input(
                "Ask a federated query:",
                placeholder="e.g., Compare the duration of MSc programs at both universities",
                key="federated_question"
            )

        with col2:
            st.write("")
            ask_button = st.button("Federated Query", type="primary", use_container_width=True)

        # Action buttons
        col3, col4 = st.columns(2)
        with col3:
            clear_history = st.button("Clear History", use_container_width=True)
        with col4:
            example_question = st.button("Example Query", use_container_width=True)

        # Handle actions
        if example_question:
            st.session_state.example_question = "Compare the duration and cost of MSc programs at Hull and Keele universities"
            st.rerun()

        if clear_history:
            st.session_state.chat_history = []
            st.rerun()

        # Use example question if set
        if hasattr(st.session_state, 'example_question'):
            user_question = st.session_state.example_question
            del st.session_state.example_question
            ask_button = True

        # Process federated query
        if ask_button and user_question:
            with st.spinner('Processing federated query across independent nodes...'):
                # Show the federated process
                st.markdown("### Federated Query Process:")

                # Step 1: Query distribution
                st.markdown("**Step 1:** Distributing query to all federated nodes...")
                time.sleep(1)

                # Step 2: Independent retrieval
                st.markdown("**Step 2:** Each node performing independent local retrieval...")
                time.sleep(1)

                # Step 3: Result aggregation
                st.markdown("**Step 3:** Aggregating results from all nodes...")
                time.sleep(1)

                # Execute federated query
                response = st.session_state.coordinator.federated_query(user_question)

                # Step 4: Generation
                st.markdown("**Step 4:** Generating federated response...")
                time.sleep(1)

                # Store in chat history
                st.session_state.chat_history.append(response)

                st.success("Federated query completed!")
                st.rerun()

if __name__ == "__main__":
    main()
