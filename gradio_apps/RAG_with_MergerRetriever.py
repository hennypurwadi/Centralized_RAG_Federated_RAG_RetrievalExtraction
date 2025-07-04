import gradio as gr
import os
import sys
from bs4 import BeautifulSoup
import string
import requests
from urllib.parse import urljoin, urlparse
import time

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
from langchain.retrievers import MergerRetriever

custom_css = """
.gradio-container {
    max-width: 1200px !important;
    margin: auto !important;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

.main-header {
    text-align: center;
    margin-bottom: 2rem;
    padding: 2rem;
    background: linear-gradient(135deg, #bae6fd 0%, #7dd3fc 100%);
    border-radius: 15px;
    color: #0c4a6e;
    box-shadow: 0 8px 32px rgba(125, 211, 252, 0.3);
}

.main-title {
    font-size: 2.5rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
    line-height: 1.2;
}

.sub-title {
    font-size: 1.2rem;
    opacity: 0.9;
    margin-bottom: 0;
}

.description {
    font-size: 1rem;
    color: #1e293b;
    margin: 2rem 0;
    padding: 1.5rem;
    background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
    border-radius: 12px;
    border-left: 5px solid #16a34a;
    box-shadow: 0 4px 16px rgba(34, 197, 94, 0.1);
}

.chat-container {
    background: white;
    padding: 1.5rem;
    border-radius: 12px;
    margin: 1rem 0;
    box-shadow: 0 4px 16px rgba(8, 145, 178, 0.1);
    border-left: 4px solid #16a34a;
}

.gr-button-primary {
    background: linear-gradient(135deg, #dc2626 0%, #ef4444 100%) !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 12px 24px !important;
    transition: all 0.3s ease !important;
    color: white !important;
}

.gr-button-primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(220, 38, 38, 0.4) !important;
}

#initialize-button {
    background: linear-gradient(135deg, #dc2626 0%, #ef4444 100%) !important;
    color: white !important;
    border: none !important;
    font-weight: bold !important;
    border-radius: 8px !important;
    padding: 12px 24px !important;
}

#initialize-button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 15px rgba(220, 38, 38, 0.4) !important;
}

.info-box {
    background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
    padding: 1.5rem;
    border-radius: 12px;
    border-left: 4px solid #3b82f6;
    margin: 1rem 0;
    color: #1e40af;
}

.footer-info {
    text-align: center;
    margin-top: 2rem;
    padding: 1.5rem;
    background: linear-gradient(135deg, #f1f5f9 0%, #e2e8f0 100%);
    border-radius: 12px;
    font-size: 0.9rem;
    color: #64748b;
    box-shadow: 0 4px 16px rgba(0,0,0,0.05);
}
"""

# Helper Functions
def clean_html(raw_html: str) -> str:
    soup = BeautifulSoup(raw_html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    return "\n".join(line.strip() for line in text.splitlines() if line.strip())

def chunk_documents(docs, chunk_size=1100, chunk_overlap=200):
    splitter = CharacterTextSplitter(separator="\n", chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = []
    for doc in docs:
        pieces = splitter.split_text(doc.page_content)
        for piece in pieces:
            chunks.append(Document(page_content=piece, metadata=doc.metadata))
    return chunks

def load_text_file(file_path, source_name):
    """Load content from a text file and create a LangChain Document."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()

        # Create document with metadata
        doc = Document(
            page_content=content,
            metadata={"source": file_path, "title": source_name}
        )
        print(f"Loaded {len(content)} characters from {file_path}", file=sys.stderr)
        return [doc]  # Return as list for consistency

    except Exception as e:
        print(f"Error loading {file_path}: {e}", file=sys.stderr)
        return []

def get_federated_docs_from_files(file_paths_and_names):
    all_docs = []
    for file_path, source_name in file_paths_and_names:
        docs = load_text_file(file_path, source_name)
        all_docs.extend(docs)
    return all_docs

# Global
qa_chain = None

def initialize_rag_system(api_key, embeddings_model, llm_model, temperature, k_docs, progress=gr.Progress()):
    global qa_chain
    if not api_key:
        return "Please enter your OpenAI API key to continue.", None
    os.environ["OPENAI_API_KEY"] = api_key
    progress(0, desc="Initializing RAG with MergerRetriever System...")

    embeddings = OpenAIEmbeddings(model=embeddings_model)

    hull_files = [
        ("./data/msc_ai_hullonline_short.txt", "Hull MSc AI Online")
    ]
    keele_files = [
        ("./data/msc_ai_keeleonline.txt", "Keele MSc Computer Science with AI")
    ]

    progress(0.1, desc="Loading Hull documents...")
    hull_docs = get_federated_docs_from_files(hull_files)
    hull_chunks = chunk_documents(hull_docs)
    progress(0.4, desc="Loading Keele documents...")
    keele_docs = get_federated_docs_from_files(keele_files)
    keele_chunks = chunk_documents(keele_docs)
    progress(0.7, desc="Building indexes...")

    hull_vectorstore = FAISS.from_documents(hull_chunks, embeddings)
    keele_vectorstore = FAISS.from_documents(keele_chunks, embeddings)

    hull_retriever = hull_vectorstore.as_retriever(search_kwargs={"k": k_docs})
    keele_retriever = keele_vectorstore.as_retriever(search_kwargs={"k": k_docs})

    merger_retriever = MergerRetriever(retrievers=[
        hull_retriever,
        keele_retriever
    ])

    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model_name=llm_model, temperature=temperature),
        chain_type="stuff",
        retriever=merger_retriever,
        return_source_documents=True
    )

    progress(1.0, desc="Initialization complete!")
    return "System Ready", qa_chain

def ask_question(question, chat_history):
    if qa_chain is None:
        return chat_history + [[question, "Please initialize the RAG system first."]]
    result = qa_chain({"query": question})
    return chat_history + [[question, result["result"]]]

# UI
with gr.Blocks(css=custom_css, theme=gr.themes.Soft(), title="RAG with MergerRetriever Chatbot") as demo:
    gr.HTML("""
        <div class="main-header">
            <h1 class="main-title">RAG with MergerRetriever Chatbot</h1>
            <p class="sub-title">RETRIEVAL-AUGMENTED GENERATION WITH MERGERRETRIEVER</p>
            <p style="font-size: 1rem; opacity: 0.8;">Compares MSc Artificial Intelligence online at University of Hull & MSc Computer Science with Artificial Intelligence online at Keele University</p>
        </div>
    """)

    gr.HTML("""
        <div class="description">
            <p><strong>How it works:</strong> This RAG system uses a MergerRetriever to combine information from multiple universities
            to provide comprehensive comparisons between Hull's MSc AI and Keele's MSc Computer Science with AI programs.</p>
        </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("##Configuration")
            api_key_input = gr.Textbox(label="OpenAI API Key", type="password", value=os.environ.get("OPENAI_API_KEY", ""))
            with gr.Accordion("Model Settings", open=False):
                embeddings_model_dropdown = gr.Dropdown(["text-embedding-3-small", "text-embedding-3-large"], label="Embeddings Model", value="text-embedding-3-small")
                llm_model_dropdown = gr.Dropdown(["gpt-4o", "gpt-3.5-turbo"], label="LLM Model", value="gpt-4o")
                temperature_slider = gr.Slider(0, 1, value=0.3, step=0.1, label="Temperature")
            with gr.Accordion("Retrieval Settings", open=False):
                k_docs_slider = gr.Slider(3, 15, value=5, step=1, label="Documents per source")
            initialize_button = gr.Button("Initialize RAG with MergerRetriever System", variant="primary", size="lg", elem_id="initialize-button")
            status_output = gr.Markdown("👆 Click Initialize to start!", elem_classes=["info-box"])

        with gr.Column(scale=2):
            gr.Markdown("## Chat Interface")
            question_input = gr.Textbox(label="Enter your question:", lines=2)
            ask_button = gr.Button("Ask", variant="primary", size="lg")
            clear_button = gr.Button("Clear History", variant="secondary")
            chatbot = gr.Chatbot(label="Chat History", height=400)

    initialize_button.click(initialize_rag_system,
        inputs=[api_key_input, embeddings_model_dropdown, llm_model_dropdown, temperature_slider, k_docs_slider],
        outputs=[status_output, gr.State()]
    )

    ask_button.click(ask_question, inputs=[question_input, chatbot], outputs=chatbot)
    question_input.submit(ask_question, inputs=[question_input, chatbot], outputs=chatbot)
    clear_button.click(lambda: [], outputs=chatbot)

demo.launch()

