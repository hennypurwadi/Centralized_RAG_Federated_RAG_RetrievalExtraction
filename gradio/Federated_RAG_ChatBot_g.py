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

def scrape_website(url, max_depth=1):
    visited = set()
    documents = []
    domain = urlparse(url).netloc

    def scrape(current_url, depth):
        if depth > max_depth or current_url in visited:
            return
        visited.add(current_url)
        try:
            response = requests.get(current_url, timeout=10, headers={'User-Agent': 'Mozilla/5.0'})
            response.encoding = response.apparent_encoding
            response.raise_for_status()
            content = clean_html(response.text)
            documents.append(Document(page_content=content, metadata={"source": current_url}))
            if depth < max_depth:
                soup = BeautifulSoup(response.text, "html.parser")
                for link in soup.find_all("a", href=True):
                    full_url = urljoin(current_url, link["href"])
                    if urlparse(full_url).netloc == domain and full_url not in visited:
                        scrape(full_url, depth + 1)
        except Exception as e:
            print(f"Error scraping {current_url}: {e}")
    scrape(url, 0)
    return documents

def get_federated_docs(urls, max_depth=1):
    all_docs = []
    for url in urls:
        all_docs.extend(scrape_website(url, max_depth))
    return all_docs

# Global
qa_chain = None

def initialize_rag_system(api_key, embeddings_model, llm_model, temperature, k_docs, progress=gr.Progress()):
    global qa_chain
    if not api_key:
        return "Please enter your OpenAI API key to continue.", None
    os.environ["OPENAI_API_KEY"] = api_key
    progress(0, desc="Initializing Federated RAG System...")

    embeddings = OpenAIEmbeddings(model=embeddings_model)

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

    progress(0.1, desc="Loading Hull documents...")
    hull_chunks = chunk_documents(get_federated_docs(hull_urls, 1))
    progress(0.4, desc="Loading Keele documents...")
    keele_chunks = chunk_documents(get_federated_docs(keele_urls, 1))
    progress(0.7, desc="Building indexes...")

    hull_index = FAISS.from_documents(hull_chunks, embeddings)
    keele_index = FAISS.from_documents(keele_chunks, embeddings)
    retriever = MergerRetriever(retrievers=[
        hull_index.as_retriever(search_kwargs={"k": k_docs}),
        keele_index.as_retriever(search_kwargs={"k": k_docs})
    ])

    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model_name=llm_model, temperature=temperature),
        retriever=retriever,
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
with gr.Blocks(css=custom_css, theme=gr.themes.Soft(), title="Federated RAG Chatbot") as demo:
    gr.HTML("""
        <div class="main-header">
            <h1 class="main-title">Federated RAG Chatbot</h1>
            <p class="sub-title">FEDERATED RETRIEVAL-AUGMENTED GENERATION</p>
            <p style="font-size: 1rem; opacity: 0.8;">Compares MSc Artificial Intelligence online at University of Hull & MSc Computer Science with Artificial Intelligence online at Keele University</p>
        </div>
    """)

    gr.HTML("""
        <div class="description">
            <p><strong>How it works:</strong> This Federated RAG system combines information from multiple universities
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
            initialize_button = gr.Button("Initialize Federated RAG System", variant="primary", size="lg", elem_id="initialize-button")
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
