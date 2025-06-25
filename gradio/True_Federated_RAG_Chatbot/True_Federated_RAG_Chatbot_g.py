import gradio as gr
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

# Get OpenAI API key from Hugging Face Spaces secrets
def get_api_key():
    """Get OpenAI API key from HF Spaces secrets or environment variables"""
    # Try to get from HF Spaces secrets first
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables. Please set it in your Hugging Face Space secrets.")
    return api_key

# Initialize API key
try:
    api_key = get_api_key()
    os.environ["OPENAI_API_KEY"] = api_key
except ValueError as e:
    print(f"Error: {e}")
    sys.exit(1)

# Helper document processing functions
def clean_html(raw_html: str) -> str:
    """Clean HTML content and extract text"""
    soup = BeautifulSoup(raw_html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)

def chunk_documents(docs, chunk_size=1100, chunk_overlap=200):
    """Split documents into chunks"""
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
    """Scrape website content"""
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
            print(f"Error scraping {current_url}: {str(e)}")

    scrape(url, 0)
    return documents

def get_federated_docs(urls, max_depth=1):
    """Get documents from multiple URLs"""
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

    def initialize_node(self, k_docs: int = 5, progress_callback=None):
        """Initialize the federated node with its own private data"""
        if progress_callback:
            progress_callback(f"Initializing {self.name} node (PRIVATE)...")

        # Load and process documents locally
        raw_docs = get_federated_docs(self.urls, max_depth=1)
        self.documents = chunk_documents(raw_docs)

        # Build local vector store (PRIVATE)
        self.vector_store = FAISS.from_documents(self.documents, self.embeddings)
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": k_docs})

        self.is_initialized = True
        if progress_callback:
            progress_callback(f"{self.name} node ready ({len(self.documents)} chunks)")

    def local_retrieve(self, query: str) -> List[Dict[str, Any]]:
        """Perform local retrieval and return ONLY the results (not raw data)"""
        if not self.is_initialized:
            raise ValueError(f"{self.name} node not initialized")

        # Use the updated invoke method instead of deprecated get_relevant_documents
        local_docs = self.retriever.invoke(query)

        # Return only the results (no access to raw vector store)
        results = []
        for i, doc in enumerate(local_docs):
            results.append({
                "content": doc.page_content[:400] + "..." if len(doc.page_content) > 400 else doc.page_content,
                "source": doc.metadata.get("source", "Unknown"),
                "node": self.name,
                "rank": i + 1
            })
        return results

class FederationCoordinator:
    """Coordinates federated queries among multiple independent nodes"""

    def __init__(self, nodes: List[FederatedNode], llm_model: str = "gpt-4o", temperature: float = 0.3):
        self.nodes = nodes
        self.llm = ChatOpenAI(model_name=llm_model, temperature=temperature)

    def federated_query(self, query: str, max_response_length: int = 200, progress_callback=None) -> Dict[str, Any]:
        """Send query to all federated nodes and aggregate results"""
        all_results = []
        node_responses = {}

        # Send query to each federated node independently
        for node in self.nodes:
            if progress_callback:
                progress_callback(f"Querying {node.name} node...")
            node_results = node.local_retrieve(query)
            node_responses[node.name] = node_results
            all_results.extend(node_results)

        # Aggregate & rank results where federation happens
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

        if progress_callback:
            progress_callback("Generating federated response...")
        
        # Use the updated invoke method instead of deprecated predict
        response = self.llm.invoke(prompt).content

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

# Global variables for the federated system
hull_node = None
keele_node = None
coordinator = None
chat_history = []
system_initialized = False

# Define federated node URLs
HULL_URLS = [
    "https://online.hull.ac.uk",
    "https://online.hull.ac.uk/contact",
    "https://online.hull.ac.uk/why-join-us/faqs",
    "https://online.hull.ac.uk/funding-options",
    "https://online.hull.ac.uk/course-costs",
    "https://online.hull.ac.uk/courses/msc-artificial-intelligence"
]

KEELE_URLS = [
    "https://online.keele.ac.uk/online-programme/msc-computer-science-with-artificial-intelligence",
    "https://online.keele.ac.uk/online-study",
    "https://online.keele.ac.uk/about-us"
]

def initialize_federated_system(embeddings_model="text-embedding-3-small", llm_model="gpt-4o", temperature=0.3, k_docs=5, progress=gr.Progress()):
    """Initialize the federated RAG system with progress tracking"""
    global hull_node, keele_node, coordinator, system_initialized
    
    try:
        progress_updates = []
        
        def progress_callback(message):
            progress_updates.append(message)
            return message
        
        # Progress tracking
        progress(0.1, desc="Starting federated system initialization...")
        time.sleep(0.5)
        
        # Create independent federated nodes
        progress(0.2, desc="Creating Hull University node...")
        progress_callback("Creating Hull University node...")
        hull_node = FederatedNode("Hull University", HULL_URLS, embeddings_model)
        time.sleep(0.5)
        
        progress(0.3, desc="Creating Keele University node...")
        progress_callback("Creating Keele University node...")
        keele_node = FederatedNode("Keele University", KEELE_URLS, embeddings_model)
        time.sleep(0.5)
        
        # Initialize each node independently
        progress(0.4, desc="Initializing Hull University data...")
        hull_node.initialize_node(k_docs, progress_callback)
        progress(0.7, desc="Initializing Keele University data...")
        keele_node.initialize_node(k_docs, progress_callback)
        
        # Create federation coordinator
        progress(0.9, desc="Creating federation coordinator...")
        progress_callback("Creating federation coordinator...")
        coordinator = FederationCoordinator([hull_node, keele_node], llm_model, temperature)
        
        system_initialized = True
        progress(1.0, desc="Federated System Ready!")
        progress_callback("Federated System Ready!")
        
        return "\n".join(progress_updates), get_system_status()
        
    except Exception as e:
        return f"Error initializing system: {str(e)}", "❌ System not initialized"

def get_system_status():
    """Get current system status"""
    if not system_initialized:
        return "❌ System not initialized"
    
    status = "**Federated System Active**\n\n"
    status += "**Federated Nodes:**\n"
    status += f"• Hull University: {len(hull_node.documents)} documents\n"
    status += f"• Keele University: {len(keele_node.documents)} documents\n"
    status += f"• Total Queries: {len(chat_history)}\n"
    status += "• Privacy: Complete data separation maintained"
    
    return status

def reset_system():
    """Reset the federated system"""
    global hull_node, keele_node, coordinator, chat_history, system_initialized
    
    hull_node = None
    keele_node = None
    coordinator = None
    chat_history = []
    system_initialized = False
    
    return "System reset successfully", "❌ System not initialized"

def process_federated_query(query, max_response_length=200, progress=gr.Progress()):
    """Process a federated query with progress tracking"""
    global chat_history
    
    if not system_initialized:
        return "**Please initialize the federated system first!**\n\n Click the **'Initialize Federated System'** button above to get started.", get_chat_history()
    
    if not query.strip():
        return "Please enter a valid query", get_chat_history()
    
    try:
        progress_updates = []
        
        def progress_callback(message):
            progress_updates.append(message)
        
        # Progress tracking for query processing
        progress(0.1, desc=" Starting federated query...")
        time.sleep(0.3)
        
        progress(0.3, desc=" Distributing query to nodes...")
        time.sleep(0.3)
        
        progress(0.5, desc=" Processing at each node...")
        
        # Execute federated query
        response = coordinator.federated_query(query, max_response_length, progress_callback)
        
        progress(0.9, desc=" Formatting response...")
        time.sleep(0.3)
        
        # Add to chat history
        chat_history.append(response)
        
        # Format response for display
        formatted_response = f"** Query:** {query}\n\n"
        formatted_response += f"** Federated AI Response:** {response['answer']}\n\n"
        formatted_response += f"** Sources:** {response['total_sources']} documents from {len(response['node_responses'])} nodes\n\n"
        
        # Add node-specific results
        formatted_response += "** Detailed Results by Node:**\n"
        for node_name, results in response['node_responses'].items():
            formatted_response += f"\n**{node_name}:**\n"
            for i, result in enumerate(results[:3], 1):  # Show top 3 results
                formatted_response += f"{i}. {result['content'][:150]}...\n"
                formatted_response += f"   *Source: {result['source']}*\n"
        
        progress(1.0, desc=" Query completed!")
        
        return formatted_response, get_chat_history()
        
    except Exception as e:
        return f"❌ Error processing query: {str(e)}", get_chat_history()

def get_chat_history():
    """Get formatted chat history"""
    if not chat_history:
        return "No queries yet"
    
    history_text = ""
    for i, chat in enumerate(chat_history, 1):
        history_text += f"**Q{i}:** {chat['query']}\n"
        history_text += f"**A{i}:** {chat['answer']}\n\n"
    
    return history_text

def clear_chat_history():
    """Clear chat history"""
    global chat_history
    chat_history = []
    return "Chat history cleared", get_chat_history()

# Create Gradio interface
def create_interface():
    """Create the Gradio interface"""
    
    # Custom CSS for enhanced styling
    custom_css = """
    .initialize-btn {
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4) !important;
        border: none !important;
        color: white !important;
        font-weight: bold !important;
        font-size: 16px !important;
        padding: 12px 24px !important;
        border-radius: 8px !important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2) !important;
        transition: all 0.3s ease !important;
    }
    
    .initialize-btn:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(0,0,0,0.3) !important;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #FFE066, #FF6B6B);
        border: 2px solid #FF6B6B;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        color: #8B0000;
        font-weight: bold;
        text-align: center;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.7; }
        100% { opacity: 1; }
    }
    
    .status-active {
        background: linear-gradient(135deg, #4ECDC4, #44A08D);
        color: white;
        padding: 10px;
        border-radius: 8px;
        font-weight: bold;
    }
    
    .status-inactive {
        background: linear-gradient(135deg, #FF6B6B, #C44569);
        color: white;
        padding: 10px;
        border-radius: 8px;
        font-weight: bold;
    }
    """
    
    with gr.Blocks(title="🐭 True Federated RAG Chatbot", theme=gr.themes.Soft(), css=custom_css) as demo:
        
        # Header
        gr.Markdown("""
        # 🐭 True Federated RAG Chatbot
        
        ### MSc Artificial Intelligence online at University of Hull & MSc Computer Science with Artificial Intelligence online at Keele University
        
        **Independent University Nodes with Complete Data Privacy**
        
        Each university maintains its own private data & retrieval system. Only final results are shared, no raw data or direct access to other nodes' vector stores.
        """)
        
        with gr.Row():
            with gr.Column(scale=2):
                # Main chat interface
                gr.Markdown("## Federated Query Interface")
                
                query_input = gr.Textbox(
                    label="Ask a federated query",
                    placeholder="e.g., How long is the duration of MSc Artificial Intelligence online at University of Hull?",
                    lines=3
                )
                
                with gr.Row():
                    query_btn = gr.Button("Federated Query", variant="primary", size="lg")
                    example_btn = gr.Button("Example Query", size="lg")
                    clear_btn = gr.Button("Clear History", size="lg")
                
                response_output = gr.Markdown(value="Enter a query to get started", label="Response")
                
            with gr.Column(scale=1):
                # System controls and status
                gr.Markdown("## System Controls")
                
                # Eye-catching initialize button
                gr.HTML("""
                <div class="warning-box">
                    SYSTEM NOT INITIALIZED<br>
                    Click the button below to start!
                </div>
                """)
                
                init_btn = gr.Button(
                    "Initialize Federated System", 
                    variant="primary", 
                    size="lg",
                    elem_classes=["initialize-btn"]
                )
                
                with gr.Accordion("Model Settings", open=False):
                    embeddings_model = gr.Dropdown(
                        choices=["text-embedding-3-small", "text-embedding-3-large"],
                        value="text-embedding-3-small",
                        label="Embeddings Model"
                    )
                    llm_model = gr.Dropdown(
                        choices=["gpt-4o", "gpt-3.5-turbo"],
                        value="gpt-4o",
                        label="LLM Model"
                    )
                    temperature = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.3,
                        step=0.1,
                        label="Temperature"
                    )
                    k_docs = gr.Slider(
                        minimum=3,
                        maximum=10,
                        value=5,
                        step=1,
                        label="Documents per node"
                    )
                    max_response_length = gr.Slider(
                        minimum=100,
                        maximum=500,
                        value=200,
                        step=50,
                        label="Max response length"
                    )
                
                reset_btn = gr.Button("Reset System", size="lg")
                
                system_status = gr.Markdown(value="❌ System not initialized")
                
                init_output = gr.Textbox(
                    label="Initialization Log",
                    lines=8,
                    max_lines=8,
                    value=""
                )
        
        # Federated Architecture Display
        with gr.Row():
            with gr.Column():
                gr.Markdown("""
                ## Federated Node Architecture
                
                ### Hull University Node (PRIVATE)
                - **Local Data:** MSc AI program information
                - **Vector Store:** Private FAISS index  
                - **Access:** Results only, no raw data sharing
                - **Status:** Independent & Secure
                
                ### Keele University Node (PRIVATE)
                - **Local Data:** MSc CS with AI program information
                - **Vector Store:** Private FAISS index
                - **Access:** Results only, no raw data sharing  
                - **Status:** Independent & Secure
                
                ### Federation Coordinator
                - **Function:** Distributes queries to all nodes and aggregates ONLY the final results
                - **Privacy:** No access to raw data or vector stores from any node
                - **Process:** Query → Distribute → Local Retrieval → Result Aggregation → Generation
                """)
        
        # Chat History
        with gr.Row():
            with gr.Column():
                gr.Markdown("## Chat History")
                history_output = gr.Markdown(value="No queries yet")
        
        # Event handlers
        def handle_example():
            return "How long is the duration of MSc Artificial Intelligence online at University of Hull?"
        
        def handle_query(query, max_len):
            return process_federated_query(query, max_len)
        
        def handle_init(emb_model, llm_model_val, temp, k_docs_val):
            return initialize_federated_system(emb_model, llm_model_val, temp, k_docs_val)
        
        def handle_clear():
            return clear_chat_history()
        
        # Connect events
        example_btn.click(
            fn=handle_example,
            outputs=query_input
        )
        
        query_btn.click(
            fn=handle_query,
            inputs=[query_input, max_response_length],
            outputs=[response_output, history_output]
        )
        
        init_btn.click(
            fn=handle_init,
            inputs=[embeddings_model, llm_model, temperature, k_docs],
            outputs=[init_output, system_status]
        )
        
        reset_btn.click(
            fn=reset_system,
            outputs=[init_output, system_status]
        )
        
        clear_btn.click(
            fn=handle_clear,
            outputs=[response_output, history_output]
        )
    
    return demo

# Launch the application
if __name__ == "__main__":
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )