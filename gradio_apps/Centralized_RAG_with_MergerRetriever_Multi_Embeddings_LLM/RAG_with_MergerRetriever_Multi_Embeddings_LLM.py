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
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
from langchain.retrievers import MergerRetriever
from langchain.prompts import PromptTemplate

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

/* New styles for Markdown components */
.answer-box {
    background: linear-gradient(135deg, #f0fdfa 0%, #ccfbf1 100%);
    padding: 1.5rem;
    border-radius: 12px;
    margin: 1rem 0;
    border-left: 4px solid #0d9488;
    color: #134e4a;
    max-height: none !important;
    overflow-y: auto;
    line-height: 1.6;
}

.answer-box h3 {
    color: #0f766e;
    margin-bottom: 1rem;
    font-size: 1.2rem;
}

.chat-history {
    background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
    padding: 1.5rem;
    border-radius: 12px;
    margin: 1rem 0;
    border-left: 4px solid #64748b;
    color: #334155;
    max-height: 600px;
    overflow-y: auto;
}

.chat-history h3 {
    color: #475569;
    margin-bottom: 0.5rem;
    font-size: 1.1rem;
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
    # Load content from a text file and create a LangChain Document.
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()

        # Create doc with metadata
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

# Removed truncate_text function to allow longer responses
def create_embeddings(provider: str, embeddings_model: str, api_key: str = None):
    """Create embeddings based on provider"""
    if provider == "OpenAI":
        if not api_key:
            raise ValueError("OpenAI API key is required for OpenAI embeddings")
        os.environ["OPENAI_API_KEY"] = api_key
        return OpenAIEmbeddings(model=embeddings_model)
    elif provider == "HuggingFace (Free)":
        # Use All-MiniLM embeddings for HuggingFace
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': True}
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
    else:
        raise ValueError(f"Unsupported provider: {provider}")

def create_llm(provider: str, llm_model: str, temperature: float, api_key: str = None):
    # Create LLM based on provider
    if provider == "OpenAI":
        if not api_key:
            raise ValueError("OpenAI API key is required for OpenAI models")
        os.environ["OPENAI_API_KEY"] = api_key
        return ChatOpenAI(temperature=temperature, model_name=llm_model)
    elif provider == "HuggingFace (Free)":
        # Use HuggingFace models 
        try:
            from transformers import pipeline, AutoTokenizer
            
            # Use DialoGPT-medium for better responses
            model_name = "microsoft/DialoGPT-medium"
            
            # Create tokenizer with proper configuration
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            pipe = pipeline(
                "text-generation",
                model=model_name,
                tokenizer=tokenizer,
                max_new_tokens=1024,  
                temperature=temperature,
                do_sample=True if temperature > 0 else False,
                pad_token_id=tokenizer.eos_token_id,
                truncation=True
            )
            return HuggingFacePipeline(pipeline=pipe)
            
        except Exception as e:
            # Fallback to distilgpt2 with better configuration
            try:
                from transformers import pipeline, AutoTokenizer
                
                # Create tokenizer with proper configuration
                tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                
                pipe = pipeline(
                    "text-generation",
                    model="distilgpt2",
                    tokenizer=tokenizer,
                    max_new_tokens=512,  
                    temperature=temperature,
                    do_sample=True if temperature > 0 else False,
                    pad_token_id=tokenizer.eos_token_id,
                    truncation=True
                )
                return HuggingFacePipeline(pipeline=pipe)
            except Exception as fallback_e:
                raise ValueError(f"Failed to load HuggingFace model: {str(e)}. Fallback also failed: {str(fallback_e)}")
    else:
        raise ValueError(f"Unsupported provider: {provider}")

# Prompt template 
HUGGINGFACE_QA_PROMPT = PromptTemplate(
    template="""Based on the following context, please provide a comprehensive, detailed, and complete answer to the question. Do not truncate or shorten your response. Include all relevant information from the context.

Context:
{context}

Question: {question}

Detailed Answer (provide a complete response):""",
    input_variables=["context", "question"]
)

# Global
qa_chain = None
current_provider = None

def initialize_rag_system(provider, api_key, embeddings_model, llm_model, temperature, k_docs, progress=gr.Progress()):
    global qa_chain, current_provider
    
    # Check if API key is required
    if provider == "OpenAI" and not api_key:
        return "Please enter your OpenAI API key to continue.", None
    
    progress(0, desc="Initializing Centralized RAG System...")

    try:
        current_provider = provider
        
        # Create embeddings based on provider
        embeddings = create_embeddings(provider, embeddings_model, api_key)
        
        # Create LLM based on provider
        llm = create_llm(provider, llm_model, temperature, api_key)

        hull_files = [
            ("./data/msc_ai_hullonline_short.txt", "Hull MSc AI Online")
        ]
        keele_files = [
            ("./data/msc_ai_keeleonline_short.txt", "Keele MSc Computer Science with AI")
        ]

        progress(0.1, desc="Loading Hull documents...")
        hull_docs = get_federated_docs_from_files(hull_files)
        # Use reasonable chunks for all models
        chunk_size = 800 if provider == "HuggingFace (Free)" else 1100
        hull_chunks = chunk_documents(hull_docs, chunk_size=chunk_size)
        
        progress(0.4, desc="Loading Keele documents...")
        keele_docs = get_federated_docs_from_files(keele_files)
        keele_chunks = chunk_documents(keele_docs, chunk_size=chunk_size)
        
        progress(0.7, desc="Building indexes...")

        hull_vectorstore = FAISS.from_documents(hull_chunks, embeddings)
        keele_vectorstore = FAISS.from_documents(keele_chunks, embeddings)

        # Use reasonable k_docs for all models
        effective_k_docs = min(k_docs, 8) if provider == "HuggingFace (Free)" else k_docs
        
        hull_retriever = hull_vectorstore.as_retriever(search_kwargs={"k": effective_k_docs})
        keele_retriever = keele_vectorstore.as_retriever(search_kwargs={"k": effective_k_docs})

        merger_retriever = MergerRetriever(retrievers=[
            hull_retriever,
            keele_retriever
        ])

        # Use different prompt templates based on provider
        chain_type_kwargs = {}
        if provider == "HuggingFace (Free)":
            chain_type_kwargs = {"prompt": HUGGINGFACE_QA_PROMPT}

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=merger_retriever,
            return_source_documents=True,
            chain_type_kwargs=chain_type_kwargs
        )

        progress(1.0, desc="Initialization complete!")
        return f"System Ready with {provider} provider", qa_chain
        
    except Exception as e:
        return f"Error initializing system: {str(e)}", None

def ask_question(question, chat_history):
    global qa_chain, current_provider
    
    if qa_chain is None:
        return chat_history + [[question, "Please initialize the RAG system first."]]
    
    try:
        # For HuggingFace models, use improved processing
        if current_provider == "HuggingFace (Free)":
            # Get relevant documents
            docs = qa_chain.retriever.get_relevant_documents(question)
            
            # Build context without any content limitations
            context_parts = []
            for doc in docs[:8]:  # Use top 8 documents for better context
                # Use full document content without character limits
                context_parts.append(doc.page_content)
            
            context = "\n\n".join(context_parts)
            
            # Create comprehensive prompt without truncation
            prompt = f"""Based on the following context, please provide a comprehensive, detailed, and complete answer to the question. Do not truncate or shorten your response. Include all relevant information from the context.

Context:
{context}

Question: {question}

Detailed Answer (provide a complete response):"""
            
            try:
                result = qa_chain.llm(prompt)
                # Clean up the response
                if isinstance(result, list) and len(result) > 0:
                    answer = result[0]['generated_text']
                    # Extract only the answer part
                    if "Detailed Answer" in answer:
                        answer = answer.split("Detailed Answer")[1]
                        if ":" in answer:
                            answer = answer.split(":", 1)[1].strip()
                        else:
                            answer = answer.strip()
                    elif "Answer:" in answer:
                        answer = answer.split("Answer:")[-1].strip()
                elif isinstance(result, str):
                    answer = result
                else:
                    answer = str(result)
                
                # Clean up the answer without removing content
                answer = answer.replace(prompt, "").strip()
                
                # Ensure we have a meaningful response
                if not answer or len(answer) < 20:
                    # Provide a more informative fallback that includes context
                    answer = f"I found relevant information about your question. Here's what I can tell you based on the available documents:\n\n{context[:1000]}..."
                    
            except Exception as e:
                # Provide more informative fallback with full context
                answer = f"I found relevant information about your question. Here's what I can tell you based on the available documents:\n\n{context[:1000]}..."
        else:
            # Use normal processing for OpenAI
            result = qa_chain({"query": question})
            answer = result["result"]
        
        return chat_history + [[question, answer]]
        
    except Exception as e:
        error_msg = f"Error processing question: {str(e)}\n\nThis might be due to token limitations. Try asking a more specific question or initialize with OpenAI for better results."
        return chat_history + [[question, error_msg]]

def update_model_options(provider):
    """Update model options based on selected provider"""
    if provider == "OpenAI":
        embeddings_options = ["text-embedding-3-small", "text-embedding-3-large"]
        llm_options = ["gpt-4o", "gpt-3.5-turbo"]
        api_key_visible = True
    elif provider == "HuggingFace (Free)":
        embeddings_options = ["all-MiniLM-L6-v2 (Free)"]
        llm_options = ["DialoGPT-medium (Free)", "DistilGPT2 (Free)"]
        api_key_visible = False
    else:
        embeddings_options = []
        llm_options = []
        api_key_visible = True
    
    return (
        gr.Dropdown(choices=embeddings_options, value=embeddings_options[0] if embeddings_options else None),
        gr.Dropdown(choices=llm_options, value=llm_options[0] if llm_options else None),
        gr.Textbox(visible=api_key_visible)
    )

# UI
with gr.Blocks(css=custom_css, theme=gr.themes.Soft(), title="CENTRALIZED RAG Chatbot") as demo:
    gr.HTML("""
        <div class="main-header">
            <h1 class="main-title">Centralized RAG Chatbot</h1>
            <p class="sub-title">Centralized RAG with Merger Retriever</p>
            <p style="font-size: 1rem; opacity: 0.8;">Compares MSc Artificial Intelligence online at University of Hull & MSc Computer Science with Artificial Intelligence online at Keele University</p>
        </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## Configuration")
            
            # Provider selection
            provider_dropdown = gr.Dropdown(
                ["OpenAI", "HuggingFace (Free)"], 
                label="Provider", 
                value="OpenAI",
                info="Choose between OpenAI or Huggingface - LLM will follow this selection. Note: OpenAI generally performs better."
            )
            
            api_key_input = gr.Textbox(
                label="OpenAI API Key", 
                type="password", 
                value=os.environ.get("OPENAI_API_KEY", ""),
                info="Required only for OpenAI provider"
            )
            
            with gr.Accordion("Model Settings", open=False):
                embeddings_model_dropdown = gr.Dropdown(
                    ["text-embedding-3-small", "text-embedding-3-large"], 
                    label="Embeddings Model", 
                    value="text-embedding-3-small"
                )
                llm_model_dropdown = gr.Dropdown(
                    ["gpt-4o", "gpt-3.5-turbo"], 
                    label="LLM Model", 
                    value="gpt-4o"
                )
                temperature_slider = gr.Slider(0, 1, value=0.3, step=0.1, label="Temperature")
            
            with gr.Accordion("Retrieval Settings", open=False):
                k_docs_slider = gr.Slider(3, 15, value=5, step=1, label="Documents per source")
            
            initialize_button = gr.Button("Initialize Centralized RAG System", variant="primary", size="lg", elem_id="initialize-button")
            status_output = gr.Markdown("👆 Click Initialize to start!", elem_classes=["info-box"])

        with gr.Column(scale=2):
            gr.Markdown("## Chat Interface")
            question_input = gr.Textbox(label="Enter your question:", lines=2, placeholder="Ask about MSc AI programs at Hull or Keele...")
            
            with gr.Row():
                ask_button = gr.Button("Ask", variant="primary", size="lg")
                example_button = gr.Button("Example Question", variant="secondary")
                clear_button = gr.Button("Clear History", variant="secondary")
            
            # Replace Chatbot with Markdown for better display of long responses
            answer_output = gr.Markdown(label="Answer", elem_classes=["answer-box"])
            
            # Add chat history display
            with gr.Accordion("Chat History", open=True):
                chat_history_md = gr.Markdown(elem_classes=["chat-history"])

    # Sample questions section
    gr.HTML("""
        <div class="info-box">
            <h3>Sample Questions:</h3>
            <ol>
                <li>Compare the total program costs between MSc Artificial Intelligence at University of Hull online and Keele University online.</li>
                <li>What are the entry requirement differences between MSc Artificial Intelligence at University of Hull online and Keele University?</li>
                <li>Which program between MSc Artificial Intelligence at University of Hull online and Keele University online offers more flexibility for working professionals?</li>
                <li>Compare the technical skills and programming languages covered between MSc Artificial Intelligence at University of Hull online and Keele University online.</li>
                <li>How do the start dates and program durations differ between MSc Artificial Intelligence at University of Hull online and Keele University?</li>
                <li>What are the assessment method differences between MSc Artificial Intelligence at University of Hull online and Keele University online?</li>
                <li>Which program is better suited for career changers between MSc Artificial Intelligence at University of Hull online and Keele University online?</li>
            </ol>
        </div>
    """)

    # Update model options when provider changes
    provider_dropdown.change(
        update_model_options,
        inputs=[provider_dropdown],
        outputs=[embeddings_model_dropdown, llm_model_dropdown, api_key_input]
    )

    initialize_button.click(initialize_rag_system,
        inputs=[provider_dropdown, api_key_input, embeddings_model_dropdown, llm_model_dropdown, temperature_slider, k_docs_slider],
        outputs=[status_output, gr.State()]
    )

    # Create a new function to handle question asking with Markdown output
    def process_question_for_markdown(question):
        global qa_chain, current_provider
        
        if qa_chain is None:
            return "Please initialize the RAG system first.", "No chat history yet."
        
        try:
            # For HuggingFace models, use improved processing
            if current_provider == "HuggingFace (Free)":
                # Get relevant documents
                docs = qa_chain.retriever.get_relevant_documents(question)
                
                # Build context without any content limitations
                context_parts = []
                for doc in docs[:8]:  # Use top 8 documents for better context
                    # Use full document content without character limits
                    context_parts.append(doc.page_content)
                
                context = "\n\n".join(context_parts)
                
                # Create comprehensive prompt without truncation
                prompt = f"""Based on the following context, please provide a comprehensive, detailed, and complete answer to the question. Do not truncate or shorten your response. Include all relevant information from the context.

Context:
{context}

Question: {question}

Detailed Answer (provide a complete response):"""
                
                try:
                    result = qa_chain.llm(prompt)
                    # Clean up the response
                    if isinstance(result, list) and len(result) > 0:
                        answer = result[0]['generated_text']
                        # Extract only the answer part
                        if "Detailed Answer" in answer:
                            answer = answer.split("Detailed Answer")[1]
                            if ":" in answer:
                                answer = answer.split(":", 1)[1].strip()
                            else:
                                answer = answer.strip()
                        elif "Answer:" in answer:
                            answer = answer.split("Answer:")[-1].strip()
                    elif isinstance(result, str):
                        answer = result
                    else:
                        answer = str(result)
                    
                    # Clean up the answer without removing content
                    answer = answer.replace(prompt, "").strip()
                    
                    # Ensure we have a meaningful response
                    if not answer or len(answer) < 20:
                        # Provide a more informative fallback that includes context
                        answer = f"I found relevant information about your question. Here's what I can tell you based on the available documents:\n\n{context[:1000]}..."
                        
                except Exception as e:
                    # Provide more informative fallback with full context
                    answer = f"I found relevant information about your question. Here's what I can tell you based on the available documents:\n\n{context[:1000]}..."
            else:
                # Use normal processing for OpenAI
                result = qa_chain({"query": question})
                answer = result["result"]
            
            # Update chat history
            global chat_history
            if not hasattr(process_question_for_markdown, 'chat_history'):
                process_question_for_markdown.chat_history = []
            
            process_question_for_markdown.chat_history.append((question, answer))
            
            # Format chat history for display
            chat_history_text = ""
            for i, (q, a) in enumerate(process_question_for_markdown.chat_history, 1):
                chat_history_text += f"### Question {i}:\n{q}\n\n### Answer {i}:\n{a}\n\n---\n\n"
            
            return f"### Your Question:\n{question}\n\n### Answer:\n{answer}", chat_history_text
            
        except Exception as e:
            error_msg = f"Error processing question: {str(e)}\n\nThis might be due to token limitations. Try asking a more specific question or initialize with OpenAI for better results."
            return error_msg, "Error occurred during processing."
    
    # Clear chat history function
    def clear_chat_history():
        if hasattr(process_question_for_markdown, 'chat_history'):
            process_question_for_markdown.chat_history = []
        return "", "Chat history cleared."
    
    # Function to get example questions
    def get_example_question():
        examples = [
            "Compare the total program costs between MSc Artificial Intelligence at University of Hull online and Keele University online.",
            "What are the entry requirement differences between MSc Artificial Intelligence at University of Hull online and Keele University?",
            "Which program between MSc Artificial Intelligence at University of Hull online and Keele University online offers more flexibility for working professionals?",
            "Compare the technical skills and programming languages covered between MSc Artificial Intelligence at University of Hull online and Keele University online.",
            "How do the start dates and program durations differ between MSc Artificial Intelligence at University of Hull online and Keele University?",
            "What are the assessment method differences between MSc Artificial Intelligence at University of Hull online and Keele University online?",
            "Which program is better suited for career changers between MSc Artificial Intelligence at University of Hull online and Keele University online?"
        ]
        import random
        return random.choice(examples)
    
    # Connect the new function to the UI
    ask_button.click(
        process_question_for_markdown,
        inputs=[question_input],
        outputs=[answer_output, chat_history_md]
    )
    
    question_input.submit(
        process_question_for_markdown,
        inputs=[question_input],
        outputs=[answer_output, chat_history_md]
    )
    
    clear_button.click(
        clear_chat_history,
        outputs=[answer_output, chat_history_md]
    )
    
    example_button.click(
        get_example_question,
        outputs=[question_input]
    )

demo.launch()

