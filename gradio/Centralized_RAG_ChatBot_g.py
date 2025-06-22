import gradio as gr
import os
import string
import time
from typing import List, Tuple

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader

# Global variables to store the RAG system components
vector_store = None
qa_chain = None
initialized = False
chat_history = []

# Custom CSS for bluish theme
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
    background: linear-gradient(135deg, #0891b2 0%, #1e40af 100%);
    border-radius: 15px;
    color: white;
    box-shadow: 0 8px 32px rgba(8, 145, 178, 0.3);
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
    border-left: 5px solid #0891b2;
    box-shadow: 0 4px 16px rgba(8, 145, 178, 0.1);
}

.chat-container {
    background: white;
    padding: 1.5rem;
    border-radius: 12px;
    margin: 1rem 0;
    box-shadow: 0 4px 16px rgba(8, 145, 178, 0.1);
    border-left: 4px solid #0891b2;
}

.question-text {
    font-weight: bold;
    color: #1e40af;
    margin-bottom: 0.5rem;
    font-size: 1.1rem;
}

.answer-text {
    color: #0f766e;
    line-height: 1.6;
    background: linear-gradient(135deg, #f0fdfa 0%, #ccfbf1 100%);
    padding: 1rem;
    border-radius: 8px;
    margin-top: 0.5rem;
}

.source-container {
    background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
    padding: 1rem;
    border-radius: 8px;
    margin: 0.5rem 0;
    border-left: 3px solid #64748b;
}

.info-box {
    background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
    padding: 1.5rem;
    border-radius: 12px;
    border-left: 4px solid #3b82f6;
    margin: 1rem 0;
    color: #1e40af;
}

.stats-container {
    background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
    padding: 1.5rem;
    border-radius: 12px;
    margin: 1rem 0;
    border: 1px solid #0891b2;
}

.gr-button-primary {
    background: linear-gradient(135deg, #0891b2 0%, #1e40af 100%) !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 12px 24px !important;
    transition: all 0.3s ease !important;
    color: white !important;
}

.gr-button-primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(8, 145, 178, 0.4) !important;
}

.gr-button-secondary {
    background: linear-gradient(135deg, #64748b 0%, #475569 100%) !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    color: white !important;
}

.gr-textbox {
    border-radius: 8px !important;
    border: 2px solid #e2e8f0 !important;
    transition: all 0.3s ease !important;
}

.gr-textbox:focus {
    border-color: #0891b2 !important;
    box-shadow: 0 0 0 3px rgba(8, 145, 178, 0.1) !important;
}

.gr-slider {
    accent-color: #0891b2 !important;
}

.gr-dropdown {
    border-radius: 8px !important;
    border: 2px solid #e2e8f0 !important;
}

.gr-accordion {
    border-radius: 8px !important;
    border: 1px solid #e2e8f0 !important;
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

def load_data(file_path: str):
    """Load text data from file"""
    try:
        text_loader = TextLoader(file_path=file_path)
        return text_loader.load()
    except Exception as e:
        return None

def initialize_rag_system(api_key: str, embeddings_model: str, llm_model: str, 
                         temperature: float, k_docs: int, chunk_size: int, 
                         chunk_overlap: int, data_file_path: str, progress=gr.Progress()):
    """Initialize the RAG system with given parameters"""
    global vector_store, qa_chain, initialized
    
    if not api_key:
        return "❌ Please provide an OpenAI API key", "", ""
    
    try:
        # Set API key
        os.environ["OPENAI_API_KEY"] = api_key
        
        progress(0.1, desc="Setting up embeddings...")
        # Initialize embedding model
        embeddings = OpenAIEmbeddings(model=embeddings_model)
        
        progress(0.2, desc="Loading documents...")
        # Load documents
        documents = load_data(data_file_path)
        if not documents:
            return "❌ Failed to load documents. Check file path.", "", ""
        
        progress(0.4, desc="Processing documents...")
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", "!", "?", ";", ":", " ", ""],
            length_function=len
        )
        split_docs = text_splitter.split_documents(documents)
        
        progress(0.6, desc="Building FAISS index...")
        # Build FAISS index
        vector_store = FAISS.from_documents(split_docs, embeddings)
        
        progress(0.8, desc="Creating QA chain...")
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
        
        initialized = True
        progress(1.0, desc="Complete!")
        
        return (f"RAG System initialized successfully!\n\n"
                f"Processed {len(split_docs)} document chunks\n"
                f"Using {llm_model} with {embeddings_model}\n"
                f"Retrieving top {k_docs} relevant documents"), "", ""
        
    except Exception as e:
        return f"❌ Error initializing RAG system: {str(e)}", "", ""

def ask_question(question: str, progress=gr.Progress()):
    """Process a question through the RAG system"""
    global qa_chain, initialized, chat_history
    
    if not initialized or not qa_chain:
        return "❌ Please initialize the RAG system first.", "", ""
    
    if not question.strip():
        return "⚠️ Please enter a question.", "", ""
    
    try:
        progress(0.3, desc="Retrieving relevant documents...")
        # Get response from QA chain
        response = qa_chain({"query": question})
        answer = response["result"]
        sources = response["source_documents"]
        
        progress(0.8, desc="Generating response...")
        
        # Format sources
        sources_text = ""
        for i, source in enumerate(sources, 1):
            source_file = source.metadata.get('source', 'Unknown')
            content_preview = source.page_content[:300] + "..." if len(source.page_content) > 300 else source.page_content
            sources_text += f"**Source {i}:** {source_file}\n\n{content_preview}\n\n---\n\n"
        
        # Add to chat history
        chat_history.append((question, answer, sources))
        
        # Format chat history for display
        chat_display = ""
        for i, (q, a, _) in enumerate(chat_history, 1):
            chat_display += f"**Q{i}:** {q}\n\n**AI:** {a}\n\n---\n\n"
        
        progress(1.0, desc="Complete!")
        
        return answer, sources_text, chat_display
        
    except Exception as e:
        return f"❌ Error processing question: {str(e)}", "", ""

def clear_chat_history():
    """Clear the chat history"""
    global chat_history
    chat_history = []
    return "", "", ""

def get_example_question():
    """Return an example question"""
    examples = [
        "How long does the MSc Artificial Intelligence course take to complete?",
        "What is the total cost of the MSc Artificial Intelligence Online program?",
        "What is the minimum academic qualification required to apply?",
        "What language proficiency is required if my first language isn't English?",
        "What if I don't have a degree but have relevant professional experience?"
    ]
    import random
    return random.choice(examples)

def get_statistics():
    """Get chat statistics"""
    global chat_history
    if not chat_history:
        return "No questions asked yet."
    
    total_questions = len(chat_history)
    total_sources = sum(len(sources) for _, _, sources in chat_history)
    avg_sources = total_sources / total_questions if total_questions > 0 else 0
    
    return f"""📊 **Session Statistics**

🔢 **Total Questions:** {total_questions}
📚 **Total Sources Used:** {total_sources}
📈 **Average Sources per Question:** {avg_sources:.1f}
"""

# Create the Gradio interface
with gr.Blocks(css=custom_css, title="Centralized RAG Chatbot", theme=gr.themes.Soft()) as demo:
    # Header section
    gr.HTML("""
        <div class="main-header">
            <h1 class="main-title">Centralized RAG Chatbot</h1>
            <p class="sub-title">RETRIEVAL-AUGMENTED GENERATION</p>
            <p style="font-size: 1rem; opacity: 0.8;">Uses centralized approach to retrieve MSc AI program information and generates natural responses with advanced AI</p>
        </div>
    """)
    
    # Description section
    gr.HTML("""
        <div class="description">
            <p><strong>ℹ️ How it works:</strong> This RAG (Retrieval-Augmented Generation) system combines document retrieval
            with AI generation to provide accurate, contextual answers about the MSc AI Online program at the University of Hull.</p>
        </div>
    """)
    
    with gr.Row():
        # Left column - Configuration
        with gr.Column(scale=1):
            gr.Markdown("## ⚙️ Configuration")
            
            # API Key
            api_key = gr.Textbox(
                label="🔑 OpenAI API Key",
                type="password",
                placeholder="Enter your OpenAI API key...",
                value=os.environ.get("OPENAI_API_KEY", "")
            )
            
            # Model Settings
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
                    value=0.0,
                    step=0.1,
                    label="Temperature"
                )
            
            # Retrieval Settings
            with gr.Accordion("🔍 Retrieval Settings", open=False):
                k_docs = gr.Slider(
                    minimum=3,
                    maximum=15,
                    value=5,
                    step=1,
                    label="Documents to retrieve"
                )
            
            # Document Processing
            with gr.Accordion("📄 Document Processing", open=False):
                chunk_size = gr.Slider(
                    minimum=300,
                    maximum=1000,
                    value=500,
                    step=50,
                    label="Chunk Size"
                )
                
                chunk_overlap = gr.Slider(
                    minimum=50,
                    maximum=200,
                    value=100,
                    step=10,
                    label="Chunk Overlap"
                )
                
                data_file_path = gr.Textbox(
                    label="Data File Path",
                    value="./data/msc_ai_hullonline_short.txt",
                    placeholder="Path to text data file"
                )
            
            # Initialize button
            init_btn = gr.Button(
                "🚀 Initialize RAG System",
                variant="primary",
                size="lg"
            )
            
            # Status display
            status_display = gr.Markdown(
                "👆 Click Initialize to start!",
                elem_classes=["info-box"]
            )
        
        # Right column - Chat Interface
        with gr.Column(scale=2):
            gr.Markdown("## 💬 Chat Interface")
            
            # Question input
            with gr.Row():
                question_input = gr.Textbox(
                    label="Enter your question:",
                    placeholder="e.g., How long does the MSc Artificial Intelligence course take to complete?",
                    lines=2,
                    scale=4
                )
                ask_btn = gr.Button(
                    "Ask",
                    variant="primary",
                    size="lg",
                    scale=1
                )
            
            # Action buttons
            with gr.Row():
                example_btn = gr.Button("💡 Example Question", variant="secondary")
                clear_btn = gr.Button("🗑️ Clear History", variant="secondary")
                stats_btn = gr.Button("📈 Statistics", variant="secondary")
            
            # Response display
            answer_output = gr.Markdown(
                label="Answer:",
                visible=True
            )
            
            # Sources display
            with gr.Accordion("📚 Sources", open=False):
                sources_output = gr.Markdown()
            
            # Chat history
            with gr.Accordion("💬 Chat History", open=True):
                chat_output = gr.Markdown()
    
    # Sample questions section
    gr.HTML("""
        <div class="info-box">
            <h3>💡 Sample Questions You Could Ask:</h3>
            <ol>
                <li>How long does the MSc Artificial Intelligence course take to complete?</li>
                <li>What is the total cost of the MSc Artificial Intelligence Online program?</li>
                <li>What is the minimum academic qualification required to apply for the MSc Artificial Intelligence (Online) program?</li>
                <li>What if I don't have a degree but have relevant professional experience?</li>
                <li>What language proficiency is required if my first language isn't English?</li>
            </ol>
        </div>
    """)
    
    # Footer
    gr.HTML("""
        <div class="footer-info">
            <p><strong>About:</strong> MSc Artificial Intelligence Online Program - University of Hull</p>
            <p><strong>Powered by:</strong> OpenAI GPT & LangChain RAG Technology</p>
        </div>
    """)
    
    # Event handlers
    init_btn.click(
        fn=initialize_rag_system,
        inputs=[api_key, embeddings_model, llm_model, temperature, k_docs, 
                chunk_size, chunk_overlap, data_file_path],
        outputs=[status_display, sources_output, chat_output],
        show_progress=True
    )
    
    ask_btn.click(
        fn=ask_question,
        inputs=[question_input],
        outputs=[answer_output, sources_output, chat_output],
        show_progress=True
    )
    
    question_input.submit(
        fn=ask_question,
        inputs=[question_input],
        outputs=[answer_output, sources_output, chat_output],
        show_progress=True
    )
    
    example_btn.click(
        fn=get_example_question,
        outputs=[question_input]
    )
    
    clear_btn.click(
        fn=clear_chat_history,
        outputs=[answer_output, sources_output, chat_output]
    )
    
    stats_btn.click(
        fn=get_statistics,
        outputs=[answer_output]
    )

# Launch the app
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
