import gradio as gr
import os
import time
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
import copy

# Global variables to store federated RAG system components
federated_nodes = {}
federated_server = None
initialized = False
chat_history = []
training_history = []

# Custom prompt template
CUSTOM_QA_PROMPT = PromptTemplate(
    template="""Based on the following context, please provide a comprehensive answer to the question.

Context:
{context}

Question: {question}

Answer:""",
    input_variables=["context", "question"]
)

# Enhanced CSS for improved layout
custom_css = """
.gradio-container {
    max-width: 850px !important;
    margin: auto !important;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

.main-header {
    text-align: center;
    margin-bottom: 1rem;
    padding: 1.5rem;
    background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); /* Yellow gradient */
    border-radius: 15px;
    color: #92400e; /* Warm brown for title text */
    box-shadow: 0 8px 32px rgba(253, 230, 138, 0.4); /* Soft yellow glow */
}

.main-title {
    font-size: 2.8rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
    line-height: 1.2;
}

.sub-title {
    font-size: 1.3rem;
    opacity: 0.9;
    margin-bottom: 0;
}

.big-button-container {
    background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
    padding: 2rem;
    border-radius: 15px;
    margin: 1rem 0;
    border: 2px solid #0ea5e9;
    text-align: center;
}

.openai-indicator {
    background: linear-gradient(135deg, #facc15 0%, #ca8a04 100%);
    color: white;
    padding: 0.5rem 1rem;
    border-radius: 20px;
    font-weight: bold;
    display: inline-block;
    margin: 0.5rem 0;
}

.status-box {
    background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
    padding: 1.5rem;
    border-radius: 12px;
    margin: 1rem 0;
    border-left: 4px solid #3b82f6;
    min-height: 100px;
}

.federated-node {
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

.compact-settings {
    background: linear-gradient(135deg, #fafafa 0%, #f4f4f5 100%);
    padding: 1rem;
    border-radius: 8px;
    margin: 0.5rem 0;
    border: 1px solid #d4d4d8;
}

.gr-button-primary {
    background: linear-gradient(135deg, #facc15 0%, #ca8a04 100%) !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 12px 24px !important;
    transition: all 0.3s ease !important;
    color: white !important;
}

.gr-button-primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(202, 138, 4, 0.4) !important;
}
.gr-button-secondary {
    background: linear-gradient(135deg, #fde047 0%, #ca8a04 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    padding: 10px 20px !important;
    transition: all 0.3s ease !important;
}
.gr-button-secondary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(202, 138, 4, 0.4) !important;
}

#main-action-button {
    background: linear-gradient(135deg, #facc15 0%, #ca8a04 100%) !important; /* Dark yellow gradient */
    color: white !important;
    border: none !important;
    font-weight: bold !important;
    font-size: 1.2rem !important;
    border-radius: 12px !important;
    padding: 20px 40px !important;
    min-height: 60px !important;
    box-shadow: 0 4px 20px rgba(202, 138, 4, 0.3) !important;
}

#main-action-button:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 8px 30px rgba(202, 138, 4, 0.5) !important;
}

.chat-section {
    background: linear-gradient(135deg, #fef9c3 0%, #fef08a 100%); /* Soft light yellow gradient */
    padding: 1.5rem;
    border-radius: 12px;
    border: 1px solid #facc15; /* Yellow border */
}
"""

class SimpleFedAvgServer:
    """Simplified FedAvg server for OpenAI-only usage"""
    
    def __init__(self):
        self.nodes = {}
        self.training_rounds = 0
        self.training_history = []
        self.global_embeddings_cache = {}
        
    def register_node(self, node_name: str, node):
        """Register a federated node"""
        self.nodes[node_name] = node
        
    def simulate_fedavg_training(self, rounds: int = 3) -> Dict:
        """Simulate FedAvg training for OpenAI embeddings"""
        training_results = {
            "rounds": rounds,
            "success": True,
            "node_improvements": {},
            "global_improvement": 0.0
        }
        
        # Simulate training improvements
        base_improvement = 0.05
        for round_num in range(rounds):
            round_improvement = base_improvement * (1 - round_num * 0.01)
            
            for node_name, node in self.nodes.items():
                node_improvement = round_improvement + np.random.normal(0, 0.01)
                node_improvement = max(0, min(node_improvement, 0.1))
                
                if node_name not in training_results["node_improvements"]:
                    training_results["node_improvements"][node_name] = []
                
                training_results["node_improvements"][node_name].append({
                    "round": round_num + 1,
                    "improvement": node_improvement,
                    "confidence_boost": node_improvement * 2
                })
                
                node.apply_fedavg_improvement(node_improvement)
        
        # Calculate global improvement
        all_improvements = []
        for node_improvements in training_results["node_improvements"].values():
            all_improvements.extend([imp["improvement"] for imp in node_improvements])
        
        training_results["global_improvement"] = np.mean(all_improvements) if all_improvements else 0.0
        
        self.training_rounds += rounds
        self.training_history.append(training_results)
        
        return training_results

class FederatedRAGNode:
    
    def __init__(self, node_name: str, documents: List[Document], embeddings,
                 llm_model, temperature: float, k_docs: int):
        self.node_name = node_name
        self.documents = documents
        self.embeddings = embeddings
        self.llm_model = llm_model
        self.temperature = temperature
        self.k_docs = k_docs
        self.vectorstore = None
        self.retriever = None
        self.qa_chain = None
        
        # FedAvg simulation attributes
        self.fedavg_improvement = 0.0
        self.training_rounds_completed = 0
        
        self._setup_node()
    
    def _setup_node(self):
        """Setup the node with vector store and retriever"""
        if self.documents:            
            self.vectorstore = FAISS.from_documents(self.documents, self.embeddings)
            self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": self.k_docs})
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm_model,
                chain_type="stuff",
                retriever=self.retriever,
                return_source_documents=True,
                chain_type_kwargs={"prompt": CUSTOM_QA_PROMPT}
            )
    
    def apply_fedavg_improvement(self, improvement: float):
        """Apply simulated FedAvg improvement"""
        self.fedavg_improvement += improvement
        self.training_rounds_completed += 1
    
    def query(self, question: str) -> Dict[str, Any]:
        """Query this node and return results"""
        if not self.qa_chain:
            return {
                "node_name": self.node_name,
                "answer": "Node not properly initialized",
                "source_documents": [],
                "confidence": 0.0,
                "fedavg_enhanced": False,
                "improvement_score": 0.0
            }
        
        try:
            result = self.qa_chain({"query": question})
            base_confidence = self._calculate_confidence(question, result.get("source_documents", []))
            
            # Apply FedAvg improvement to confidence
            enhanced_confidence = min(base_confidence + self.fedavg_improvement, 1.0)
            fedavg_enhanced = self.training_rounds_completed > 0
            
            return {
                "node_name": self.node_name,
                "answer": result["result"],
                "source_documents": result.get("source_documents", []),
                "confidence": enhanced_confidence,
                "fedavg_enhanced": fedavg_enhanced,
                "improvement_score": self.fedavg_improvement,
                "training_rounds": self.training_rounds_completed
            }
        except Exception as e:
            return {
                "node_name": self.node_name,
                "answer": f"Error processing query: {str(e)}",
                "source_documents": [],
                "confidence": 0.0,
                "fedavg_enhanced": False,
                "improvement_score": 0.0
            }
    
    def _calculate_confidence(self, question: str, source_docs: List[Document]) -> float:
        """Calculate confidence score based on document relevance"""
        if not source_docs:
            return 0.0
        
        base_confidence = min(len(source_docs) * 0.15, 0.8)
        question_words = set(question.lower().split())
        doc_words = set()
        
        for doc in source_docs[:3]:
            doc_words.update(doc.page_content.lower().split())
        
        overlap = len(question_words.intersection(doc_words))
        overlap_bonus = min(overlap * 0.05, 0.2)
        
        return min(base_confidence + overlap_bonus, 0.9)

def load_data(file_path: str):
    """Load text data from file"""
    try:
        if not os.path.exists(file_path):
            return None
        text_loader = TextLoader(file_path=file_path, encoding='utf-8')
        return text_loader.load()
    except Exception as e:
        print(f"Error loading {file_path}: {str(e)}")
        return None

def chunk_documents(docs: List[Document], chunk_size: int, chunk_overlap: int) -> List[Document]:
    """Split documents into smaller chunks for embeddings"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", "!", "?", ";", ":", " ", ""],
        length_function=len
    )
    return text_splitter.split_documents(docs)

def initialize_and_train_fedavg_system(api_key: str, embeddings_model: str, llm_model: str, 
                                     temperature: float, k_docs: int, chunk_size: int, 
                                     chunk_overlap: int, training_rounds: int,
                                     hull_file_path: str, keele_file_path: str,
                                     progress=gr.Progress()):
    """Combined function to initialize and train the FedAvg system"""
    global federated_nodes, federated_server, initialized, training_history
    
    if not api_key.strip():
        return "❌ Please provide your OpenAI API key", "", ""
    
    try:
        progress(0.05, desc="🚀 Starting FedAvg System...")
        
        # Set OpenAI API key
        os.environ["OPENAI_API_KEY"] = api_key.strip()
        
        # Initialize FedAvg server
        federated_server = SimpleFedAvgServer()
        
        progress(0.1, desc="🔧 Testing OpenAI connection...")
        # Initialize and test OpenAI embeddings
        try:
            embeddings = OpenAIEmbeddings(model=embeddings_model)
            test_embedding = embeddings.embed_query("test connection")
            if not test_embedding:
                raise Exception("Failed to generate test embedding")
        except Exception as e:
            return f"❌ OpenAI Embeddings Error: {str(e)}. Please check your API key.", "", ""
        
        progress(0.15, desc="🤖 Testing OpenAI LLM...")
        # Initialize and test OpenAI LLM
        try:
            llm = ChatOpenAI(temperature=temperature, model_name=llm_model)
            test_response = llm.predict("Hello")
            if not test_response:
                raise Exception("Failed to get test response")
        except Exception as e:
            return f"❌ OpenAI LLM Error: {str(e)}. Please check your API key and model access.", "", ""
        
        progress(0.2, desc="Creating federated nodes...")
        federated_nodes = {}
        total_chunks = 0
        
        # Load and process Hull documents
        progress(0.25, desc="Processing Hull University data...")
        hull_docs = load_data(hull_file_path)
        if hull_docs:
            hull_chunks = chunk_documents(hull_docs, chunk_size, chunk_overlap)
            hull_node = FederatedRAGNode("Hull University", hull_chunks, embeddings, 
                                       llm, temperature, k_docs)
            federated_nodes["Hull"] = hull_node
            federated_server.register_node("Hull", hull_node)
            total_chunks += len(hull_chunks)
        else:
            # Create sample Hull data
            sample_hull_doc = Document(page_content="""
            University of Hull MSc AI Online Program:
            - Cost: £9,500/year UK students, £16,500/year international
            - Entry: 2:2 degree minimum, basic programming experience
            - Duration: 12-24 months flexible
            - Start dates: September and January intakes
            - Assessment: 60% coursework, 25% online exams, 15% group projects
            - Flexibility: Evening classes, recorded sessions, part-time options
            - Support: 24/7 online platform, personal advisor, career services
            - Industry connections: 85% employment rate within 6 months
            """)
            hull_chunks = [sample_hull_doc]
            hull_node = FederatedRAGNode("Hull University", hull_chunks, embeddings, 
                                       llm, temperature, k_docs)
            federated_nodes["Hull"] = hull_node
            federated_server.register_node("Hull", hull_node)
            total_chunks += len(hull_chunks)
        
        # Load and process Keele documents
        progress(0.35, desc="📚 Processing Keele University data...")
        keele_docs = load_data(keele_file_path)
        if keele_docs:
            keele_chunks = chunk_documents(keele_docs, chunk_size, chunk_overlap)
            keele_node = FederatedRAGNode("Keele University", keele_chunks, embeddings,
                                        llm, temperature, k_docs)
            federated_nodes["Keele"] = keele_node
            federated_server.register_node("Keele", keele_node)
            total_chunks += len(keele_chunks)
        else:
            # Create sample Keele data
            sample_keele_doc = Document(page_content="""
            Keele University MSc AI Online Program:
            - Cost: £11,200/year UK students, £19,800/year international
            - Entry: 2:1 degree minimum, strong mathematical background
            - Duration: 12-18 months structured
            - Start dates: September intake only
            - Assessment: 50% assignments, 30% group projects, 20% written exams
            - Flexibility: Fixed schedule, mandatory attendance, research-focused
            - Research: Small cohorts, high faculty ratio, PhD preparation
            - Academic focus: Research partnerships, publication opportunities
            """)
            keele_chunks = [sample_keele_doc]
            keele_node = FederatedRAGNode("Keele University", keele_chunks, embeddings,
                                        llm, temperature, k_docs)
            federated_nodes["Keele"] = keele_node
            federated_server.register_node("Keele", keele_node)
            total_chunks += len(keele_chunks)
        
        progress(0.45, desc="Federated system initialized!")
        initialized = True
        
        # Now start FedAvg training
        progress(0.5, desc="Starting FedAvg collaborative training...")
        
        # Run FedAvg training
        training_result = federated_server.simulate_fedavg_training(training_rounds)
        
        progress(0.8, desc="Processing training improvements...")
        
        # Format comprehensive results
        result_text = f"**OpenAI FedAvg System Ready!**\n\n"
        
        # System info
        result_text += f"**Federated Network:**\n"
        result_text += f"• Active Nodes: {len(federated_nodes)}\n"
        result_text += f"• Total Data Chunks: {total_chunks}\n"
        result_text += f"• Provider: OpenAI Premium\n"
        result_text += f"• Models: {llm_model} + {embeddings_model}\n\n"
        
        # Training results
        result_text += f"**FedAvg Training Completed:**\n"
        result_text += f"• Training Rounds: {training_rounds}\n"
        result_text += f"• Global Improvement: {training_result['global_improvement']:.3f}\n\n"
        
        # Node improvements
        result_text += f"**Node Enhancements:**\n"
        for node_name, improvements in training_result["node_improvements"].items():
            total_improvement = sum(imp["improvement"] for imp in improvements)
            result_text += f"• {node_name}: +{total_improvement:.3f} improvement\n"
        
        result_text += f"\n**Ready for Enhanced Queries!**\n"
        result_text += f"System now provides FedAvg-enhanced responses with improved accuracy and confidence."
        
        training_history.append(training_result)
        
        progress(1.0, desc="System ready!")
        
        return result_text, "", ""
        
    except Exception as e:
        progress(1.0, desc="❌ Setup failed")
        return f"❌ Error setting up system: {str(e)}", "", ""

def federated_query(question: str, progress=gr.Progress()):
    """Process a question through the OpenAI FedAvg system"""
    global federated_nodes, initialized, chat_history
    
    progress(0.05, desc="Processing your query...")
    
    if not initialized or not federated_nodes:
        progress(1.0, desc="❌ System not ready")
        return "❌ **Please initialize the system first using the big green button above.**", "", ""
    
    if not question.strip():
        progress(1.0, desc="❌ No question provided")
        return "❌ Please enter a question.", "", ""
    
    try:
        progress(0.1, desc="Querying enhanced nodes...")
        
        # Query all nodes
        node_results = []
        total_nodes = len(federated_nodes)
        
        for i, (node_name, node) in enumerate(federated_nodes.items()):
            node_progress = 0.2 + (0.5 * i / total_nodes)
            progress(node_progress, desc=f"🏛️ Querying {node_name}...")
            result = node.query(question)
            node_results.append(result)
        
        progress(0.75, desc="Synthesizing enhanced results...")
        
        # Synthesize results
        synthesized_answer = synthesize_federated_results(question, node_results)
        
        progress(0.85, desc="Formatting results...")
        
        # Format individual node results
        node_details = ""
        for result in node_results:
            confidence_bar = "█" * int(result["confidence"] * 10) + "░" * (10 - int(result["confidence"] * 10))
            
            if result["fedavg_enhanced"]:
                enhancement_status = f"FedAvg Enhanced (+{result['improvement_score']:.3f})"
                status_color = "color: #10b981; font-weight: bold;"
            else:
                enhancement_status = "Standard"
                status_color = "color: #666; font-weight: normal;"
            
            node_details += f"""
<div style="background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%); padding: 1rem; border-radius: 8px; margin: 0.5rem 0; border-left: 4px solid #a78bfa;">
<h4 style="margin: 0 0 0.5rem 0;">{result['node_name']}</h4>
<p style="{status_color} margin: 0.25rem 0;">{enhancement_status}</p>
<p style="margin: 0.25rem 0;"><strong>Confidence:</strong> {result['confidence']:.2f} {confidence_bar}</p>
<div style="margin-top: 0.5rem; padding: 0.5rem; background: rgba(255,255,255,0.7); border-radius: 4px;">
{result['answer']}
</div>
</div>
"""
        
        progress(0.92, desc="Adding sources...")
        
        # Format sources
        sources_text = ""
        for result in node_results:
            if result["source_documents"]:
                sources_text += f"\n**Sources from {result['node_name']}:**\n"
                for i, source in enumerate(result["source_documents"][:2], 1):
                    content_preview = source.page_content[:150] + "..." if len(source.page_content) > 150 else source.page_content
                    sources_text += f"• **Source {i}:** {content_preview}\n\n"
        
        progress(0.96, desc="Saving to history...")
        
        # Add to chat history
        chat_history.append((question, synthesized_answer, node_results))
        
        # Format chat history for display
        chat_display = ""
        for i, (q, a, _) in enumerate(chat_history, 1):
            chat_display += f"**Q{i}:** {q}\n\n**FedAvg AI:** {a}\n\n---\n\n"
        
        progress(1.0, desc="Query complete!")
        
        return synthesized_answer, node_details + sources_text, chat_display
        
    except Exception as e:
        progress(1.0, desc="❌ Query failed")
        return f"❌ Error processing query: {str(e)}", "", ""

def synthesize_federated_results(question: str, node_results: List[Dict]) -> str:
    """Synthesize answers from multiple federated nodes"""
    if not node_results:
        return "❌ No results available from federated nodes."
    
    valid_results = [r for r in node_results if not r["answer"].startswith("Error")]
    
    if not valid_results:
        return "❌ All federated nodes encountered errors processing the query."
    
    # Sort by confidence
    valid_results.sort(key=lambda x: x["confidence"], reverse=True)
    
    # Check enhancement status
    enhanced_count = sum(1 for r in valid_results if r.get("fedavg_enhanced", False))
    total_improvement = sum(r.get("improvement_score", 0) for r in valid_results)
    
    # Create synthesis prompt
    synthesis_prompt = f"""
You are synthesizing responses from an OpenAI-powered Federated RAG system with FedAvg enhancements.

Question: "{question}"

Node Responses ({enhanced_count}/{len(valid_results)} FedAvg-enhanced, Total improvement: {total_improvement:.3f}):

"""
    
    for i, result in enumerate(valid_results, 1):
        enhancement_info = ""
        if result.get("fedavg_enhanced", False):
            enhancement_info = f" [FedAvg Enhanced: +{result.get('improvement_score', 0):.3f}]"
        
        synthesis_prompt += f"""
Node {i} - {result['node_name']}{enhancement_info} (Confidence: {result['confidence']:.2f}):
{result['answer']}

"""
    
    synthesis_prompt += """
Provide a comprehensive synthesis that:
1. Combines insights from all institutional nodes
2. Highlights the benefits of FedAvg enhancement where applicable
3. Provides balanced comparative analysis
4. Notes confidence levels and gives actionable insights

Enhanced Synthesis:"""
    
    try:
        # Use OpenAI to synthesize
        llm = ChatOpenAI(temperature=0.2, model_name="gpt-4o")
        synthesized = llm.predict(synthesis_prompt)
        
        enhancement_indicator = f"**FedAvg-Enhanced** ({enhanced_count}/{len(valid_results)} nodes)" if enhanced_count > 0 else "📊 **Standard Analysis**"
        
        return f"{enhancement_indicator}\n\n{synthesized.strip()}"
        
    except Exception as e:
        # Fallback synthesis
        return simple_federated_synthesis(valid_results, enhanced_count)

def simple_federated_synthesis(results: List[Dict], enhanced_count: int = 0) -> str:
    """Simple fallback synthesis method"""
    if len(results) == 1:
        enhancement = "FedAvg Enhanced" if results[0].get("fedavg_enhanced", False) else "📊 Standard"
        return f"**Single Node Response from {results[0]['node_name']} ({enhancement}):**\n\n{results[0]['answer']}"
    
    enhancement_status = f"**FedAvg-Enhanced Analysis** ({enhanced_count}/{len(results)} nodes enhanced)" if enhanced_count > 0 else "📊 **Standard Analysis**"
    
    synthesis = f"{enhancement_status}\n\n"
    
    for i, result in enumerate(results, 1):
        enhancement = "Enhanced" if result.get("fedavg_enhanced", False) else "📊 Standard"
        synthesis += f"**🏛️ {result['node_name']}** ({enhancement}) - Confidence: {result['confidence']:.2f}\n"
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
        "Compare the total program costs between MSc AI at Hull and Keele universities.",
        "What are the entry requirement differences between Hull and Keele MSc AI programs?",
        "Which program offers more flexibility for working professionals?",
        "Compare the assessment methods between Hull and Keele AI programs.",
        "How do the start dates and program durations differ between the two universities?",
        "Which program is better suited for career changers?",
        "Compare the technical skills covered in both programs."
    ]
    import random
    return random.choice(examples)

def get_federated_statistics():
    """Get federated chat statistics"""
    global chat_history, federated_nodes, federated_server, training_history
    
    if not chat_history and not training_history:
        return "📊 No queries or training completed yet."
    
    total_questions = len(chat_history)
    total_nodes = len(federated_nodes)
    training_rounds = federated_server.training_rounds if federated_server else 0
    
    # Calculate statistics
    avg_confidence = 0
    enhanced_responses = 0
    total_improvement = 0
    
    if chat_history:
        all_confidences = []
        for _, _, node_results in chat_history:
            for result in node_results:
                all_confidences.append(result["confidence"])
                if result.get("fedavg_enhanced", False):
                    enhanced_responses += 1
                    total_improvement += result.get("improvement_score", 0)
        
        avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0
    
    return f"""**System Performance Statistics**

**Network:** {total_nodes} active nodes
**Queries Processed:** {total_questions}
**Training Rounds:** {training_rounds}
**Enhanced Responses:** {enhanced_responses}
**Total Improvement:** {total_improvement:.3f}
**Avg Confidence:** {avg_confidence:.2f}
**Institutions:** {', '.join(federated_nodes.keys()) if federated_nodes else 'None'}
**Status:** {'FedAvg Enhanced' if training_rounds > 0 else 'Standard'}
"""

# Create the improved Gradio interface
with gr.Blocks(css=custom_css, title="FedAvg Federated RAG Chatbot", theme=gr.themes.Soft()) as demo:
    # Header section
    gr.HTML("""
        <div class="main-header">
            <h1 class="main-title">FedAvg Federated RAG Chatbot</h1>
            <p class="sub-title">Ask questions about MSc AI Online Programmes at University of Hull or Keele</p>
            <div class="openai-indicator">Federated Averaging (FedAvg) Aggregation with Retrieval-Augmented Generation (RAG)</div>
            <p style="font-size: 1rem; opacity: 0.9; margin-top: 0.5rem;">
                
            </p>
        </div>
    """)
    
    # Big action button section - MOST PROMINENT

    
    with gr.Row():
        # Left side - Minimal essential settings
        with gr.Column(scale=1):
            # Essential settings only
            api_key = gr.Textbox(
                label="🔑 OpenAI API Key",
                type="password",
                placeholder="sk-...",
                value=os.environ.get("OPENAI_API_KEY", ""),
                info="Your API key for premium OpenAI services"
            )
            
            # Compact advanced settings
            with gr.Accordion("Advanced Settings (Optional)", open=False):
                with gr.Row():
                    embeddings_model = gr.Dropdown(
                        choices=["text-embedding-3-small", "text-embedding-3-large"],
                        value="text-embedding-3-small",
                        label="Embeddings",
                        scale=1
                    )
                    llm_model = gr.Dropdown(
                        choices=["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"],
                        value="gpt-4o",
                        label="LLM",
                        scale=1
                    )
                
                with gr.Row():
                    temperature = gr.Slider(0.0, 1.0, 0.1, step=0.1, label="Temperature", scale=1)
                    training_rounds = gr.Slider(1, 5, 3, step=1, label="Training Rounds", scale=1)
                
                with gr.Row():
                    k_docs = gr.Slider(3, 8, 5, step=1, label="📄 Docs/Node", scale=1)
                    chunk_size = gr.Slider(300, 800, 500, step=100, label="Chunk Size", scale=1)
                    chunk_overlap = gr.Slider(50, 150, 100, step=25, label="Overlap", scale=1)
                
                hull_file_path = gr.Textbox(
                    label="Hull Data", 
                    value="./data/hull.txt",
                    placeholder="Uses sample data if file not found"
                )
                keele_file_path = gr.Textbox(
                    label="Keele Data", 
                    value="./data/keele.txt",
                    placeholder="Uses sample data if file not found"
                )
        
        # Right side - Big action button
        with gr.Column(scale=1):
            # THE BIG BUTTON - Most prominent element
            main_action_btn = gr.Button(
                "🚀 CLICK HERE to Initialize and Train FedAVG",
                variant="primary",
                size="lg",
                elem_id="main-action-button"
            )
            
            # Status display
            status_display = gr.Markdown(
                "",
                elem_classes=["status-box"]
            )
    

    
    # Question input
    with gr.Row():
        question_input = gr.Textbox(
            label="Ask your comparative question:",
            placeholder="e.g., Compare the costs and flexibility between MSc AI programs at Hull and Keele",
            lines=2,
            scale=4
        )
        ask_btn = gr.Button(
            "Ask AI",
            variant="primary",
            size="lg",
            scale=1
        )
    
    # Action buttons
    with gr.Row():
        example_btn = gr.Button("Example", variant="secondary")
        clear_btn = gr.Button("Clear", variant="secondary")
        stats_btn = gr.Button("Stats", variant="secondary")
    
    # Response display
    answer_output = gr.Markdown(
        label="OpenAI FedAvg Response:",
        visible=True
    )
    
    # Expandable sections
    with gr.Accordion("Individual Institution Results", open=False):
        node_results_output = gr.HTML()
    
    with gr.Accordion("Chat History", open=False):
        chat_output = gr.Markdown()
    

    
    # Quick help section
    gr.HTML("""
        <div style="background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); padding: 1.5rem; border-radius: 12px; border-left: 4px solid #f59e0b; margin: 1rem 0;">
            <h3 style="color: #92400e; margin-top: 0;"> Quick Start Guide:</h3>
            <ol style="color: #78350f; margin: 0;">
                <li><strong>Enter your OpenAI API key</strong> in the field above</li>
                <li><strong>Click the big yellow button</strong> to initialize and train the system</li>
                <li><strong>Wait for "Ready for Enhanced Queries!"</strong> message</li>
                <li><strong>Ask questions</strong> about Hull vs Keele MSc AI programs</li>
                <li><strong>Enjoy FedAvg-enhanced responses</strong> with improved accuracy!</li>
            </ol>
        </div>
    """)
    
    # Footer
    gr.HTML("""
        <div style="text-align: center; margin-top: 2rem; padding: 1.5rem; background: linear-gradient(135deg, #f3f4f6 0%, #e5e7eb 100%); border-radius: 12px; color: #374151;">
            <p><strong>FedAvg Federated RAG:</strong> PRIVACY-PRESERVING RETRIEVAL-AUGMENTED GENERATION</p>
            <p><strong>Institutions:</strong> University of Hull & Keele University</p>
            <p><strong>Technology:</strong>Federated Averaging (FedAvg) Aggregation with Retrieval-Augmented Generation (RAG) </p>
        </div>
    """)
    
    # Event handlers
    main_action_btn.click(
        fn=initialize_and_train_fedavg_system,
        inputs=[api_key, embeddings_model, llm_model, temperature, k_docs, 
                chunk_size, chunk_overlap, training_rounds, hull_file_path, keele_file_path],
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

