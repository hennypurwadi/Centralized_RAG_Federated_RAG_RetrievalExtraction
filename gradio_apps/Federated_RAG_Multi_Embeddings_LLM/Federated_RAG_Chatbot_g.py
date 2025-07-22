import gradio as gr
import os
import string
import time
from typing import List, Tuple, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate

# Global variables to store federated RAG system components
federated_nodes = {}
initialized = False
chat_history = []

# Custom prompt template without system instructions
CUSTOM_QA_PROMPT = PromptTemplate(
    template="""Based on the following context, please provide a comprehensive answer to the question.

Context:
{context}

Question: {question}

Answer:""",
    input_variables=["context", "question"]
)

# Custom CSS for bluish theme with federated styling
custom_css = """
.gradio-container {
    max-width: 1200px !important;
    margin: auto !important;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

.main-header {
    text-align: center;
    margin-bottom: 1rem;
    padding: 1rem;
    background: linear-gradient(135deg, #e0cffc 0%, #d2b7f7 100%);
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
    margin: 1rem 0;
    padding: 1rem;
}

.federated-node {
    background: linear-gradient(135deg, #f3e8ff 0%, #e9d5ff 100%);
    padding: 1rem;
    border-radius: 8px;
    margin: 0.5rem 0;
    border-left: 4px solid #a78bfa;
}

.node-result {
    background: linear-gradient(135deg, #f3e8ff 0%, #e9d5ff 100%);
    padding: 1rem;
    border-radius: 8px;
    margin: 0.5rem 0;
    border-left: 4px solid #a78bfa;
}

.confidence-score {
    background: linear-gradient(135deg, #ede9fe 0%, #ddd6fe 100%);
    padding: 0.5rem;
    border-radius: 6px;
    margin: 0.25rem 0;
    border-left: 3px solid #8b5cf6;
    font-weight: bold;
}

.chat-container {
    background: linear-gradient(135deg, #f3e8ff 0%, #e9d5ff 100%);
    padding: 1.5rem;
    border-radius: 12px;
    margin: 1rem 0;
    box-shadow: 0 4px 16px rgba(167, 139, 250, 0.1);
    border-left: 4px solid #a78bfa;
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
    background: linear-gradient(135deg, #f3e8ff 0%, #e9d5ff 100%);
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
    background: linear-gradient(135deg, #f5e6ff 0%, #e8d2f7 100%);
    padding: 1.5rem;
    border-radius: 12px;
    border-left: 4px solid #a78bfa;
    margin: 0.5rem 0;
}

.stats-container {
    background: linear-gradient(135deg, #f5e6ff 0%, #e8d2f7 100%);
}

.gr-button-primary {
    background: linear-gradient(135deg, #a78bfa 0%, #c4b5fd 100%) !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 12px 24px !important;
    transition: all 0.3s ease !important;
    color: white !important;
}

.gr-button-primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(167, 139, 250, 0.4) !important;
}

.gr-button-secondary {
    background: linear-gradient(135deg, #d8b4fe 0%, #c084fc 100%) !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    color: white !important;
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

.gr-textbox {
    border-radius: 8px !important;
    border: 2px solid #e2e8f0 !important;
    transition: all 0.3s ease !important;
}

.gr-textbox:focus {
    border-color: #a78bfa !important;
    box-shadow: 0 0 0 3px rgba(167, 139, 250, 0.1) !important;
}

.gr-slider {
    accent-color: #a78bfa !important;
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
    background: linear-gradient(135deg, #f3e8ff 0%, #e9d5ff 100%);
    border-radius: 12px;
    font-size: 0.9rem;
    color: #64748b;
     box-shadow: 0 4px 16px rgba(0,0,0,0.05);
}
"""


class FederatedRAGNode:
    def __init__(self, node_name: str, documents: List[Document], embeddings,
                 llm_model, temperature: float, k_docs: int, provider: str):
        self.node_name = node_name
        self.documents = documents
        self.embeddings = embeddings
        self.llm_model = llm_model
        self.temperature = temperature
        self.k_docs = k_docs
        self.provider = provider
        self.vectorstore = None
        self.retriever = None
        self.qa_chain = None
        self._setup_node()
    
    def _setup_node(self):
        # Setup the node with vector store and retriever
        if self.documents:            
            self.vectorstore = FAISS.from_documents(self.documents, self.embeddings) # Create vector store            
            self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": self.k_docs}) # Create retriever
            # Create QA chain with custom prompt
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm_model,
                chain_type="stuff",
                retriever=self.retriever,
                return_source_documents=True,
                chain_type_kwargs={"prompt": CUSTOM_QA_PROMPT}
            )
    
    def query(self, question: str) -> Dict[str, Any]:
        # Query this node, return results
        if not self.qa_chain:
            return {
                "node_name": self.node_name,
                "answer": "Node not properly initialized",
                "source_documents": [],
                "confidence": 0.0
            }
        
        try:
            result = self.qa_chain({"query": question})
            
            # Calculate confidence based on source document relevance
            confidence = self._calculate_confidence(question, result.get("source_documents", []))
            
            return {
                "node_name": self.node_name,
                "answer": result["result"],
                "source_documents": result.get("source_documents", []),
                "confidence": confidence
            }
        except Exception as e:
            return {
                "node_name": self.node_name,
                "answer": f"Error processing query: {str(e)}",
                "source_documents": [],
                "confidence": 0.0
            }
    
    def _calculate_confidence(self, question: str, source_docs: List[Document]) -> float:
        # Calculate confidence score based on document relevance
        if not source_docs:
            return 0.0
        
        # Confidence calculation based on document count and content overlap
        base_confidence = min(len(source_docs) * 0.2, 1.0)
        
        # Check for keyword overlap
        question_words = set(question.lower().split())
        doc_words = set()
        for doc in source_docs[:3]:  # Check top 3 documents
            doc_words.update(doc.page_content.lower().split())
        
        overlap = len(question_words.intersection(doc_words))
        overlap_bonus = min(overlap * 0.1, 0.3)
        
        return min(base_confidence + overlap_bonus, 1.0)

def load_data(file_path: str):
    # Load text data from file
    try:
        text_loader = TextLoader(file_path=file_path)
        return text_loader.load()
    except Exception as e:
        return None

def chunk_documents(docs: List[Document], chunk_size: int, chunk_overlap: int) -> List[Document]:
    # Split documents into smaller chunks for embeddings
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", "!", "?", ";", ":", " ", ""],
        length_function=len
    )
    return text_splitter.split_documents(docs)

def create_embeddings(provider: str, embeddings_model: str, api_key: str = None):
    # Create embeddings based on provider
    if provider == "OpenAI":
        if not api_key:
            raise ValueError("OpenAI API key is required for OpenAI embeddings")
        os.environ["OPENAI_API_KEY"] = api_key
        return OpenAIEmbeddings(model=embeddings_model)
    elif provider == "Huggingface":
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
    elif provider == "Huggingface":
        # Use HuggingFace pipeline for free models
        try:
            from transformers import pipeline
            pipe = pipeline(
                "text-generation",
                model="microsoft/DialoGPT-medium",
                max_new_tokens=256,
                temperature=temperature,
                do_sample=True if temperature > 0 else False
            )
            return HuggingFacePipeline(pipeline=pipe)
        except Exception as e:
            # Fallback to a basic text generation
            try:
                from transformers import pipeline
                pipe = pipeline(
                    "text-generation",
                    model="distilgpt2",
                    max_new_tokens=128,
                    temperature=temperature,
                    do_sample=True if temperature > 0 else False
                )
                return HuggingFacePipeline(pipeline=pipe)
            except Exception as fallback_e:
                raise ValueError(f"Failed to load Huggingface model: {str(e)}. Fallback also failed: {str(fallback_e)}")
    else:
        raise ValueError(f"Unsupported provider: {provider}")

def initialize_federated_rag_system(provider: str, api_key: str, embeddings_model: str, llm_model: str, 
                                  temperature: float, k_docs: int, chunk_size: int, 
                                  chunk_overlap: int, hull_file_path: str, keele_file_path: str,
                                  progress=gr.Progress()):
    # Initialize federated with given parameters
    global federated_nodes, initialized
    
    if provider == "OpenAI" and not api_key:
        return "Please provide an OpenAI API key for OpenAI models", "", ""
    
    try:
        progress(0.05, desc="🚀 Starting federated system initialization...")
        
        progress(0.1, desc="Setting up embeddings model...")
        # Initialize embedding model
        embeddings = create_embeddings(provider, embeddings_model, api_key)
        
        progress(0.2, desc="Setting up LLM model...")
        # Initialize LLM model
        llm = create_llm(provider, llm_model, temperature, api_key)
        
        progress(0.25, desc="Preparing federated nodes...")
        federated_nodes = {}
        total_chunks = 0
        
        # Load and process Hull documents
        progress(0.3, desc="Loading Hull University documents...")
        hull_docs = load_data(hull_file_path)
        if hull_docs:
            progress(0.4, desc="Chunking Hull documents...")
            hull_chunks = chunk_documents(hull_docs, chunk_size, chunk_overlap)
            progress(0.5, desc="Creating Hull University node...")
            hull_node = FederatedRAGNode("Hull University", hull_chunks, embeddings, 
                                       llm, temperature, k_docs, provider)
            federated_nodes["Hull"] = hull_node
            total_chunks += len(hull_chunks)
        
        # Load and process Keele documents
        progress(0.6, desc="Loading Keele University documents...")
        keele_docs = load_data(keele_file_path)
        if keele_docs:
            progress(0.7, desc="Chunking Keele documents...")
            keele_chunks = chunk_documents(keele_docs, chunk_size, chunk_overlap)
            progress(0.8, desc="Creating Keele University node...")
            keele_node = FederatedRAGNode("Keele University", keele_chunks, embeddings,
                                        llm, temperature, k_docs, provider)
            federated_nodes["Keele"] = keele_node
            total_chunks += len(keele_chunks)
        
        progress(0.9, desc="Finalizing federated network...")
        
        if not federated_nodes:
            progress(1.0, desc="No documents loaded")
            return "Failed to load any documents. Check file paths.", "", ""
        
        progress(0.95, desc="Validating federated setup...")
        initialized = True
        
        progress(1.0, desc="Federated system ready!")
        
        node_info = "\n".join([f"• {name}: {len(node.documents)} chunks" 
                              for name, node in federated_nodes.items()])
        
        provider_info = f"**Provider:** {provider}"
        if provider == "Huggingface":
            provider_info += " (FREE)"
        
        return (f"Federated RAG System initialized successfully!\n\n"
                f"**Active Nodes:** {len(federated_nodes)}\n"
                f"{node_info}\n"
                f"**Total Chunks:** {total_chunks}\n"
                f"{provider_info}\n"
                f"**Model:** {llm_model} with {embeddings_model}\n"
                f"**Retrieving:** top {k_docs} documents per node"), "", ""
        
    except Exception as e:
        progress(1.0, desc="Initialization failed")
        return f"Error initializing federated RAG system: {str(e)}", "", ""

def federated_query(question: str, progress=gr.Progress()):
    #Process a question through the federated RAG system
    global federated_nodes, initialized, chat_history
    
    # Show progress immediately when function is called
    progress(0.05, desc="Processing your request...")
    
    if not initialized or not federated_nodes:
        progress(0.5, desc="System not initialized")
        progress(1.0, desc="Please initialize the system first")
        return "<span style=\"color: red; font-weight: bold;\">Please initialize the Federated RAG system first.</span>", "", ""
    
    if not question.strip():
        progress(0.5, desc="No question provided")
        progress(1.0, desc="Question required")
        return "Please enter a question.", "", ""
    
    try:
        progress(0.1, desc="Preparing federated query...")
        
        progress(0.15, desc="Connecting to federated nodes...")
        
        # Query all nodes
        node_results = []
        total_nodes = len(federated_nodes)
        
        for i, (node_name, node) in enumerate(federated_nodes.items()):
            node_progress = 0.2 + (0.5 * i / total_nodes)
            progress(node_progress, desc=f"Querying {node_name} node...")
            result = node.query(question)
            node_results.append(result)
            
            # Show completion for each node
            completion_progress = 0.2 + (0.5 * (i + 1) / total_nodes)
            progress(completion_progress, desc=f" {node_name} query complete")
        
        progress(0.75, desc="Synthesizing federated results...")
        
        # Synthesize results
        synthesized_answer = synthesize_federated_results(question, node_results)
        
        progress(0.85, desc="Calculating confidence scores...")
        
        # Format individual node results
        node_details = ""
        for result in node_results:
            confidence_bar = "" * int(result["confidence"] * 10) + "░" * (10 - int(result["confidence"] * 10))
            node_details += f"""
** {result['node_name']}** (Confidence: {result['confidence']:.2f})
{confidence_bar}

{result['answer']}

---
"""       
        progress(0.92, desc=" Formatting source documents...")
        
        # Format sources
        sources_text = ""
        for result in node_results:
            if result["source_documents"]:
                sources_text += f"\n** Sources from {result['node_name']}:**\n"
                for i, source in enumerate(result["source_documents"][:3], 1):
                    content_preview = source.page_content[:200] + "..." if len(source.page_content) > 200 else source.page_content
                    sources_text += f"• **Source {i}:** {content_preview}\n\n"
        
        progress(0.96, desc=" Updating chat history...")
        
        # Add to chat history
        chat_history.append((question, synthesized_answer, node_results))
        
        # Format chat history for display
        chat_display = ""
        for i, (q, a, _) in enumerate(chat_history, 1):
            chat_display += f"**Q{i}:** {q}\n\n** Federated AI:** {a}\n\n---\n\n"
        
        progress(1.0, desc="Federated analysis complete!")
        
        return synthesized_answer, node_details + sources_text, chat_display
        
    except Exception as e:
        progress(1.0, desc=" Error occurred")
        return f"Error processing federated query: {str(e)}", "", ""

def synthesize_federated_results(question: str, node_results: List[Dict]) -> str:
    # Synthesize answers from multiple federated nodes
    if not node_results:
        return "No results available from federated nodes."
    
    valid_results = [r for r in node_results if not r["answer"].startswith("Error")] 
    
    if not valid_results:
        return "All federated nodes encountered errors processing the query."
    
    # Sort by confidence
    valid_results.sort(key=lambda x: x["confidence"], reverse=True)
    
    # Get the provider from the first node (all nodes use the same provider)
    provider = federated_nodes[list(federated_nodes.keys())[0]].provider if federated_nodes else "Unknown"
    
    # Create synthesis prompt for federated approach
    synthesis_prompt = f"""
Synthesizing responses from a FEDERATED RAG system where each university maintains complete data privacy.

Question: "{question}"

Federated Node Responses:
"""    
    for i, result in enumerate(valid_results, 1):
        synthesis_prompt += f"""
Node {i} - {result['node_name']} (Confidence: {result['confidence']:.2f}):
{result['answer']}

"""    
    synthesis_prompt += """
Provide a comprehensive federated synthesis that:
1. Combines insights from all participating institutions
2. Highlights comparative advantages and differences
3. Maintains institutional privacy while enabling informed comparisons
4. Provides balanced, objective analysis
5. Notes when information is institution-specific vs. comparative

Federated Synthesis:"""
    
    try:
        if provider == "OpenAI":
            # Use OpenAI to synthesize
            llm = ChatOpenAI(temperature=0.3, model_name="gpt-4o")
            synthesized = llm.predict(synthesis_prompt)
            return f" **Federated Analysis:**\n\n{synthesized.strip()}"
        else:
            # For Huggingface, use simple synthesis to avoid complex prompting
            return simple_federated_synthesis(valid_results)
    except Exception as e:
        # Fallback to simple synthesis
        return simple_federated_synthesis(valid_results)

def simple_federated_synthesis(results: List[Dict]) -> str:
    """Simple fallback synthesis method for federated results"""
    if len(results) == 1:
        return f" **Single Node Response from {results[0]['node_name']}:**\n\n{results[0]['answer']}"
    
    synthesis = " **Federated Analysis from Multiple Institutions:**\n\n"
    for i, result in enumerate(results, 1):
        synthesis += f"** {result['node_name']}** (Confidence: {result['confidence']:.2f}):\n"
        synthesis += f"{result['answer']}\n\n"
    
    return synthesis

def clear_chat_history():
    """Clear the chat history"""
    global chat_history
    chat_history = []
    return "", "", ""

def get_federated_example_question():
    """Return an example question for federated system"""
    examples = [
        "Compare the total program costs between MSc Artificial Intelligence at University of Hull online and Keele University online.",
        "What are the entry requirement differences between MSc Artificial Intelligence at University of Hull online and Keele University?",
        "Which program between MSc Artificial Intelligence at University of Hull online and Keele University offers more flexibility for working professionals?",
        "Compare the technical skills and programming languages covered between MSc Artificial Intelligence at University of Hull online and Keele University.",
        "How do the start dates and program durations differ between MSc Artificial Intelligence at University of Hull online and Keele University?",
        "What are the assessment method differences between MSc Artificial Intelligence at University of Hull online and Keele University?",
        "Which program is better suited for career changers between MSc Artificial Intelligence at University of Hull online and Keele University?"
    ]
    import random
    return random.choice(examples)

def get_federated_statistics():
    """Get federated chat statistics"""
    global chat_history, federated_nodes
    if not chat_history:
        return "No federated queries processed yet."
    
    total_questions = len(chat_history)
    total_nodes = len(federated_nodes)
    total_responses = sum(len(node_results) for _, _, node_results in chat_history)
    avg_confidence = 0
    
    if chat_history:
        all_confidences = []
        for _, _, node_results in chat_history:
            for result in node_results:
                all_confidences.append(result["confidence"])
        avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0
    
    provider = federated_nodes[list(federated_nodes.keys())[0]].provider if federated_nodes else "Unknown"
    
    return f""" **Federated Session Statistics**

 **Active Nodes:** {total_nodes}
 **Total Queries:** {total_questions}
 **Total Node Responses:** {total_responses}
 **Average Confidence:** {avg_confidence:.2f}
 **Participating Institutions:** {', '.join(federated_nodes.keys())}
 **Provider:** {provider}
"""

def update_models_based_on_embeddings_provider(provider):
    """Update both embeddings and LLM model choices based on embeddings provider"""
    if provider == "OpenAI":
        embeddings_choices = ["text-embedding-3-small", "text-embedding-3-large"]
        embeddings_value = "text-embedding-3-small"
        embeddings_label = "Embeddings Model"
        
        llm_choices = ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"]
        llm_value = "gpt-4o"
        llm_label = "LLM Model"
        
    elif provider == "Huggingface":
        embeddings_choices = ["sentence-transformers/all-MiniLM-L6-v2"]
        embeddings_value = "sentence-transformers/all-MiniLM-L6-v2"
        embeddings_label = "Embeddings Model (FREE - All-MiniLM)"
        
        llm_choices = [
            "microsoft/DialoGPT-medium",
            "distilgpt2"
        ]
        llm_value = "microsoft/DialoGPT-medium"
        llm_label = "LLM Model (FREE)"
    
    return (
        gr.Dropdown(
            choices=embeddings_choices,
            value=embeddings_value,
            label=embeddings_label
        ),
        gr.Dropdown(
            choices=llm_choices,
            value=llm_value,
            label=llm_label
        )
    )

def update_api_key_visibility(provider):
    """Update API key field visibility based on provider"""
    if provider == "OpenAI":
        return gr.Textbox(
            label="OpenAI API Key",
            type="password",
            placeholder="Enter your OpenAI API key...",
            value=os.environ.get("OPENAI_API_KEY", ""),
            visible=True
        )
    elif provider == "Huggingface":
        return gr.Textbox(
            label="API Key (Not Required for Free Models)",
            type="password",
            placeholder="No API key needed for Huggingface free models",
            value="",
            visible=False
        )

def federated_query_with_loading(question: str, progress=gr.Progress()):
    """Wrapper function that handles loading state for federated query"""
    # This function will be called by the button click
    return federated_query(question, progress)

# Create the Gradio interface
with gr.Blocks(css=custom_css, title="Federated RAG Chatbot", theme=gr.themes.Soft()) as demo:
    # Header section
    gr.HTML("""
        <div class="main-header">
            <h1 class="main-title">Federated RAG Chatbot</h1>
            <p class="sub-title">PRIVACY-PRESERVING RETRIEVAL-AUGMENTED GENERATION</p>
            <p style="font-size: 1rem; opacity: 0.8;">Compare online MSc AI between University of Hull and Keele University while maintaining institutional data privacy</p>
        </div>
    """)
    
    # Description section
    gr.HTML("""
        <div class="description">
            <p><strong>Federated Approach:</strong> This system maintains separate, isolated nodes for each institution's data. 
            Queries are processed independently at each node, and only synthesized results are shared to ensure the complete data privacy 
            while enabling program comparisons between Hull and Keele universities.</p>            
        </div>
    """)
    
    with gr.Row():
        # Left column - Configuration
        with gr.Column(scale=1):
            gr.Markdown("## Federated Configuration")
            
            # Embeddings Provider Selection
            provider = gr.Dropdown(
                choices=["OpenAI", "Huggingface"],
                value="OpenAI",
                label="Embeddings Provider",
                info="Choose between OpenAI or Huggingface - LLM will follow this selection. Note: OpenAI generally performs better."
            )
            
            # API Key
            api_key = gr.Textbox(
                label="OpenAI API Key",
                type="password",
                placeholder="Enter your OpenAI API key...",
                value=os.environ.get("OPENAI_API_KEY", ""),
                visible=True
            )
            
            # Model Settings
            with gr.Accordion("Model Settings", open=False):
                embeddings_model = gr.Dropdown(
                    choices=["text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"],
                    value="text-embedding-3-small",
                    label="Embeddings Model"
                )
                
                llm_model = gr.Dropdown(
                    choices=["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"],
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
            with gr.Accordion("Federated Retrieval Settings", open=False):
                k_docs = gr.Slider(
                    minimum=3,
                    maximum=15,
                    value=5,
                    step=1,
                    label="Documents per node"
                )
            
            # Document Processing
            with gr.Accordion("Node Data Processing", open=False):
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
                
                hull_file_path = gr.Textbox(
                    label="Hull Data File",
                    #value="./data/msc_ai_hullonline_short.txt",
                    value="./data/hull.txt",
                    placeholder="Path to Hull data file"
                )
                
                keele_file_path = gr.Textbox(
                    label="Keele Data File", 
                    #value="./data/msc_ai_keeleonline_short.txt",
                    value="./data/keele.txt",
                    placeholder="Path to Keele data file"
                )
            
            # Initialize button
            init_btn = gr.Button(
                "🚀 Initialize Federated System",
                variant="primary",
                size="lg"
            )
            
            # Status display
            status_display = gr.Markdown(
                "👆 Click Initialize to start the federated system!",
                elem_classes=["info-box"]
            )
        
        # Right column - Chat Interface
        with gr.Column(scale=2):
            gr.Markdown("## Federated Chat Interface")
            
            # Question input
            with gr.Row():
                question_input = gr.Textbox(
                    label="Enter your comparative question:",
                    placeholder="e.g., Compare the costs and flexibility between MSc Artificial Intelligence online at University of Hull and Keele University MSc AI online programs",
                    lines=2,
                    scale=4
                )
                ask_btn = gr.Button(
                    " Ask Federated",
                    variant="primary",
                    size="lg",
                    scale=1
                )
            
            # Action buttons
            with gr.Row():
                example_btn = gr.Button("Example Question", variant="secondary")
                clear_btn = gr.Button("Clear History", variant="secondary")
                stats_btn = gr.Button("Statistics", variant="secondary")
            
            # Response display
            answer_output = gr.Markdown(
                label="Federated Synthesis:",
                visible=True,
                elem_classes=["answer-box"]
            )            
            # Node results display
            with gr.Accordion("Individual Node Results", open=False):
                node_results_output = gr.Markdown()
            
            # Chat history
            with gr.Accordion("Federated Chat History", open=True):
                chat_output = gr.Markdown(elem_classes=["chat-history"])
    
    # Sample questions section
    gr.HTML("""
        <div class="info-box">
            <h3> Sample Federated Questions:</h3>
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
    
    # Footer
    gr.HTML("""
        <div class="footer-info">
            <p><strong> Federated RAG:</strong> Privacy-preserving comparison of AI Master's programs</p>
            <p><strong> Institutions:</strong> University of Hull & Keele University</p>
            <p><strong> Powered by:</strong> OpenAI GPT & Huggingface (FREE) + LangChain Federated Architecture</p>
        </div>
    """)
    
    # Event handlers for provider switching
    provider.change(
        fn=update_api_key_visibility,
        inputs=[provider],
        outputs=[api_key]
    )
    
    provider.change(
        fn=update_models_based_on_embeddings_provider,
        inputs=[provider],
        outputs=[embeddings_model, llm_model]
    )
    
    # Event handlers
    init_btn.click(
        fn=initialize_federated_rag_system,
        inputs=[provider, api_key, embeddings_model, llm_model, temperature, k_docs, 
                chunk_size, chunk_overlap, hull_file_path, keele_file_path],
        outputs=[status_display, node_results_output, chat_output],
        show_progress=True
    )
    
    ask_btn.click(
        fn=federated_query,
        inputs=[question_input],
        outputs=[answer_output, node_results_output, chat_output],
        show_progress=True
    )
    
    question_input.submit(
        fn=federated_query,
        inputs=[question_input],
        outputs=[answer_output, node_results_output, chat_output],
        show_progress=True
    )
    
    example_btn.click(
        fn=get_federated_example_question,
        outputs=[question_input]
    )
    
    clear_btn.click(
        fn=clear_chat_history,
        outputs=[answer_output, node_results_output, chat_output]
    )
    
    stats_btn.click(
        fn=get_federated_statistics,
        outputs=[status_display]
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        share=False,
        show_error=True
    )
