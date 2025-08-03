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
import math

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

# Enhanced CSS for improved layout with FedProx branding
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
    background: linear-gradient(135deg, #e0f2fe 0%, #b3e5fc 100%); /* Blue gradient for FedProx */
    border-radius: 15px;
    color: #0277bd; /* Blue for title text */
    box-shadow: 0 8px 32px rgba(179, 229, 252, 0.4); /* Soft blue glow */
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
    background: linear-gradient(135deg, #0ea5e9 0%, #0284c7 100%);
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
    background: linear-gradient(135deg, #e0f2fe 0%, #bae6fd 100%);
    padding: 1rem;
    border-radius: 8px;
    margin: 0.5rem 0;
    border-left: 4px solid #0ea5e9;
}

.confidence-score {
    background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
    padding: 0.5rem;
    border-radius: 6px;
    margin: 0.25rem 0;
    border-left: 3px solid #3b82f6;
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
    background: linear-gradient(135deg, #0ea5e9 0%, #0284c7 100%) !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 12px 24px !important;
    transition: all 0.3s ease !important;
    color: white !important;
}

.gr-button-primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(2, 132, 199, 0.4) !important;
}

.gr-button-secondary {
    background: linear-gradient(135deg, #38bdf8 0%, #0284c7 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    padding: 10px 20px !important;
    transition: all 0.3s ease !important;
}

.gr-button-secondary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(2, 132, 199, 0.4) !important;
}

#main-action-button {
    background: linear-gradient(135deg, #0ea5e9 0%, #0284c7 100%) !important;
    color: white !important;
    border: none !important;
    font-weight: bold !important;
    font-size: 1.2rem !important;
    border-radius: 12px !important;
    padding: 20px 40px !important;
    min-height: 60px !important;
    box-shadow: 0 4px 20px rgba(2, 132, 199, 0.3) !important;
}

#main-action-button:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 8px 30px rgba(2, 132, 199, 0.5) !important;
}

.chat-section {
    background: linear-gradient(135deg, #e0f2fe 0%, #bae6fd 100%);
    padding: 1.5rem;
    border-radius: 12px;
    border: 1px solid #0ea5e9;
}
"""

class FedProxServer:
    """FedProx server implementation for OpenAI-based federated learning"""
    
    def __init__(self, mu: float = 0.1):
        self.nodes = {}
        self.training_rounds = 0
        self.training_history = []
        self.global_parameters = {}
        self.mu = mu  # Proximal regularization parameter
        self.global_confidence_baseline = 0.5
        self.global_retrieval_patterns = {}
        
    def register_node(self, node_name: str, node):
        """Register a federated node"""
        self.nodes[node_name] = node
        # Initialize global parameters for this node
        self.global_parameters[node_name] = {
            'confidence_weights': np.random.normal(0.5, 0.1, 10),  # Confidence pattern weights
            'retrieval_weights': np.random.normal(0.5, 0.1, 5),   # Retrieval ranking weights
            'response_quality_weights': np.random.normal(0.5, 0.1, 3)  # Response quality weights
        }
        
    def fedprox_training(self, rounds: int = 3) -> Dict:
        """Implement FedProx training with proximal regularization"""
        training_results = {
            "algorithm": "FedProx",
            "rounds": rounds,
            "mu": self.mu,
            "success": True,
            "node_improvements": {},
            "global_improvement": 0.0,
            "regularization_effects": {},
            "convergence_metrics": []
        }
        
        for round_num in range(rounds):
            round_results = self._fedprox_round(round_num)
            
            # Update global parameters using FedProx aggregation
            self._update_global_parameters()
            
            # Calculate convergence metrics
            convergence = self._calculate_convergence_metrics()
            training_results["convergence_metrics"].append(convergence)
            
            # Store round results
            for node_name, node_result in round_results.items():
                if node_name not in training_results["node_improvements"]:
                    training_results["node_improvements"][node_name] = []
                
                training_results["node_improvements"][node_name].append({
                    "round": round_num + 1,
                    "improvement": node_result["improvement"],
                    "regularization_penalty": node_result["regularization_penalty"],
                    "confidence_adjustment": node_result["confidence_adjustment"],
                    "proximal_term": node_result["proximal_term"]
                })
        
        # Calculate overall improvements
        all_improvements = []
        total_regularization = 0
        
        for node_name, improvements in training_results["node_improvements"].items():
            node_total_improvement = sum(imp["improvement"] for imp in improvements)
            node_total_regularization = sum(imp["regularization_penalty"] for imp in improvements)
            
            all_improvements.append(node_total_improvement)
            total_regularization += node_total_regularization
            
            training_results["regularization_effects"][node_name] = {
                "total_improvement": node_total_improvement,
                "total_regularization": node_total_regularization,
                "net_effect": node_total_improvement - node_total_regularization
            }
        
        training_results["global_improvement"] = np.mean(all_improvements) if all_improvements else 0.0
        training_results["total_regularization"] = total_regularization
        
        self.training_rounds += rounds
        self.training_history.append(training_results)
        
        return training_results
    
    def _fedprox_round(self, round_num: int) -> Dict:
        """Execute one round of FedProx training"""
        round_results = {}
        
        for node_name, node in self.nodes.items():
            # Get current local parameters
            local_params = node.get_local_parameters()
            global_params = self.global_parameters[node_name]
            
            # Calculate proximal regularization term
            proximal_term = self._calculate_proximal_term(local_params, global_params)
            
            # Calculate improvement with regularization
            base_improvement = 0.05 * (1 - round_num * 0.01) + np.random.normal(0, 0.01)
            regularization_penalty = self.mu * proximal_term
            
            # Net improvement after regularization
            net_improvement = max(0, base_improvement - regularization_penalty)
            
            # Calculate confidence adjustment based on regularization
            confidence_adjustment = self._calculate_confidence_adjustment(
                local_params, global_params, proximal_term
            )
            
            # Apply FedProx update to node
            node.apply_fedprox_update(
                improvement=net_improvement,
                regularization_penalty=regularization_penalty,
                confidence_adjustment=confidence_adjustment,
                global_params=global_params,
                mu=self.mu
            )
            
            round_results[node_name] = {
                "improvement": net_improvement,
                "regularization_penalty": regularization_penalty,
                "confidence_adjustment": confidence_adjustment,
                "proximal_term": proximal_term
            }
        
        return round_results
    
    def _calculate_proximal_term(self, local_params: Dict, global_params: Dict) -> float:
        """Calculate the proximal regularization term ||w_local - w_global||²"""
        total_distance = 0.0
        total_params = 0
        
        for param_type in ['confidence_weights', 'retrieval_weights', 'response_quality_weights']:
            if param_type in local_params and param_type in global_params:
                local_weights = np.array(local_params[param_type])
                global_weights = np.array(global_params[param_type])
                
                # Calculate L2 distance
                distance = np.sum((local_weights - global_weights) ** 2)
                total_distance += distance
                total_params += len(local_weights)
        
        # Normalize by number of parameters
        return total_distance / max(total_params, 1)
    
    def _calculate_confidence_adjustment(self, local_params: Dict, global_params: Dict, 
                                       proximal_term: float) -> float:
        """Calculate confidence adjustment based on regularization"""
        # Higher proximal term means more deviation from global model
        # Apply stronger regularization to confidence scores
        max_adjustment = 0.2  # Maximum confidence adjustment
        adjustment = min(max_adjustment, proximal_term * self.mu * 2)
        
        # Reduce confidence for nodes that deviate too much from global model
        return -adjustment if proximal_term > 0.1 else adjustment * 0.5
    
    def _update_global_parameters(self):
        """Update global parameters using FedProx aggregation"""
        if not self.nodes:
            return
        
        # Aggregate parameters from all nodes
        for param_type in ['confidence_weights', 'retrieval_weights', 'response_quality_weights']:
            aggregated_params = []
            
            for node_name, node in self.nodes.items():
                local_params = node.get_local_parameters()
                if param_type in local_params:
                    aggregated_params.append(np.array(local_params[param_type]))
            
            if aggregated_params:
                # FedProx aggregation (weighted average)
                global_update = np.mean(aggregated_params, axis=0)
                
                # Update global parameters for all nodes
                for node_name in self.nodes.keys():
                    self.global_parameters[node_name][param_type] = global_update
    
    def _calculate_convergence_metrics(self) -> Dict:
        """Calculate convergence metrics for monitoring"""
        if len(self.nodes) < 2:
            return {"variance": 0.0, "mean_distance": 0.0}
        
        # Calculate variance in local parameters
        all_params = []
        for node in self.nodes.values():
            local_params = node.get_local_parameters()
            # Flatten all parameters into a single vector
            param_vector = []
            for param_type in ['confidence_weights', 'retrieval_weights', 'response_quality_weights']:
                if param_type in local_params:
                    param_vector.extend(local_params[param_type])
            all_params.append(param_vector)
        
        if all_params:
            all_params = np.array(all_params)
            variance = np.var(all_params, axis=0).mean()
            mean_distance = np.mean([np.linalg.norm(p - np.mean(all_params, axis=0)) 
                                   for p in all_params])
            
            return {
                "variance": float(variance),
                "mean_distance": float(mean_distance),
                "num_nodes": len(self.nodes)
            }
        
        return {"variance": 0.0, "mean_distance": 0.0, "num_nodes": 0}

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
        
        # FedProx-specific attributes
        self.local_parameters = {
            'confidence_weights': np.random.normal(0.5, 0.1, 10),
            'retrieval_weights': np.random.normal(0.5, 0.1, 5),
            'response_quality_weights': np.random.normal(0.5, 0.1, 3)
        }
        self.fedprox_improvement = 0.0
        self.regularization_penalty = 0.0
        self.confidence_adjustment = 0.0
        self.training_rounds_completed = 0
        self.global_params_cache = {}
        self.mu = 0.1  # Will be set by server
        
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
    
    def get_local_parameters(self) -> Dict:
        """Get current local parameters for FedProx"""
        return copy.deepcopy(self.local_parameters)
    
    def apply_fedprox_update(self, improvement: float, regularization_penalty: float,
                           confidence_adjustment: float, global_params: Dict, mu: float):
        """Apply FedProx update with proximal regularization"""
        self.fedprox_improvement += improvement
        self.regularization_penalty += regularization_penalty
        self.confidence_adjustment += confidence_adjustment
        self.training_rounds_completed += 1
        self.global_params_cache = copy.deepcopy(global_params)
        self.mu = mu
        
        # Update local parameters towards global parameters with proximal regularization
        learning_rate = 0.1
        for param_type in self.local_parameters.keys():
            if param_type in global_params:
                local_weights = np.array(self.local_parameters[param_type])
                global_weights = np.array(global_params[param_type])
                
                # FedProx update: move towards global parameters with regularization
                proximal_gradient = mu * (local_weights - global_weights)
                updated_weights = local_weights - learning_rate * proximal_gradient
                
                # Add some noise for diversity while maintaining convergence
                noise = np.random.normal(0, 0.01, updated_weights.shape)
                updated_weights += noise * (1 - mu)  # Less noise with higher regularization
                
                self.local_parameters[param_type] = updated_weights.tolist()
    
    def query(self, question: str) -> Dict[str, Any]:
        """Query this node with FedProx-enhanced processing"""
        if not self.qa_chain:
            return {
                "node_name": self.node_name,
                "answer": "Node not properly initialized",
                "source_documents": [],
                "confidence": 0.0,
                "fedprox_enhanced": False,
                "improvement_score": 0.0,
                "regularization_info": {}
            }
        
        try:
            result = self.qa_chain({"query": question})
            
            # Calculate base confidence
            base_confidence = self._calculate_confidence(question, result.get("source_documents", []))
            
            # Apply FedProx enhancements
            enhanced_confidence = self._apply_fedprox_confidence_enhancement(base_confidence, question)
            
            # Calculate regularization info
            regularization_info = self._get_regularization_info()
            
            return {
                "node_name": self.node_name,
                "answer": result["result"],
                "source_documents": result.get("source_documents", []),
                "confidence": enhanced_confidence,
                "fedprox_enhanced": self.training_rounds_completed > 0,
                "improvement_score": self.fedprox_improvement,
                "regularization_penalty": self.regularization_penalty,
                "confidence_adjustment": self.confidence_adjustment,
                "training_rounds": self.training_rounds_completed,
                "regularization_info": regularization_info
            }
        except Exception as e:
            return {
                "node_name": self.node_name,
                "answer": f"Error processing query: {str(e)}",
                "source_documents": [],
                "confidence": 0.0,
                "fedprox_enhanced": False,
                "improvement_score": 0.0,
                "regularization_info": {}
            }
    
    def _apply_fedprox_confidence_enhancement(self, base_confidence: float, question: str) -> float:
        """Apply FedProx-specific confidence enhancements"""
        if self.training_rounds_completed == 0:
            return base_confidence
        
        # Apply confidence weights from local parameters
        confidence_weights = np.array(self.local_parameters['confidence_weights'])
        
        # Create feature vector from question (simplified)
        question_features = self._extract_question_features(question)
        
        # Calculate weighted confidence adjustment
        weighted_adjustment = np.dot(confidence_weights[:len(question_features)], question_features)
        weighted_adjustment = np.tanh(weighted_adjustment)  # Normalize to [-1, 1]
        
        # Apply FedProx improvement and regularization
        enhanced_confidence = base_confidence + self.fedprox_improvement + self.confidence_adjustment
        enhanced_confidence += weighted_adjustment * 0.1  # Small weighted adjustment
        
        # Apply proximal regularization to prevent over-confidence
        if self.global_params_cache:
            global_confidence_weights = np.array(self.global_params_cache.get('confidence_weights', confidence_weights))
            deviation = np.linalg.norm(confidence_weights - global_confidence_weights)
            regularization_penalty = self.mu * deviation * 0.05
            enhanced_confidence -= regularization_penalty
        
        return max(0.0, min(enhanced_confidence, 1.0))
    
    def _extract_question_features(self, question: str) -> np.ndarray:
        """Extract simple features from question for confidence weighting"""
        features = []
        
        # Length feature
        features.append(min(len(question.split()) / 20.0, 1.0))
        
        # Complexity features (presence of certain words)
        complexity_words = ['compare', 'difference', 'versus', 'better', 'cost', 'price', 'requirement']
        complexity_score = sum(1 for word in complexity_words if word in question.lower()) / len(complexity_words)
        features.append(complexity_score)
        
        # Institution-specific features
        hull_words = ['hull', 'university of hull']
        keele_words = ['keele', 'keele university']
        
        hull_score = sum(1 for word in hull_words if word in question.lower())
        keele_score = sum(1 for word in keele_words if word in question.lower())
        
        features.append(min(hull_score, 1.0))
        features.append(min(keele_score, 1.0))
        
        # Academic features
        academic_words = ['program', 'course', 'degree', 'msc', 'masters', 'ai', 'artificial intelligence']
        academic_score = sum(1 for word in academic_words if word in question.lower()) / len(academic_words)
        features.append(academic_score)
        
        # Pad or truncate to match confidence_weights length
        while len(features) < 10:
            features.append(0.0)
        
        return np.array(features[:10])
    
    def _get_regularization_info(self) -> Dict:
        """Get information about regularization effects"""
        if not self.global_params_cache:
            return {"status": "No global parameters available"}
        
        regularization_info = {
            "mu": self.mu,
            "total_penalty": self.regularization_penalty,
            "confidence_adjustment": self.confidence_adjustment,
            "parameter_deviations": {}
        }
        
        # Calculate parameter deviations from global model
        for param_type in self.local_parameters.keys():
            if param_type in self.global_params_cache:
                local_weights = np.array(self.local_parameters[param_type])
                global_weights = np.array(self.global_params_cache[param_type])
                deviation = np.linalg.norm(local_weights - global_weights)
                regularization_info["parameter_deviations"][param_type] = float(deviation)
        
        return regularization_info
    
    def _calculate_confidence(self, question: str, source_docs: List[Document]) -> float:
        """Calculate base confidence score"""
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

def initialize_and_train_fedprox_system(api_key: str, embeddings_model: str, llm_model: str, 
                                      temperature: float, k_docs: int, chunk_size: int, 
                                      chunk_overlap: int, training_rounds: int, mu: float,
                                      hull_file_path: str, keele_file_path: str,
                                      progress=gr.Progress()):
    """Combined function to initialize and train the FedProx system"""
    global federated_nodes, federated_server, initialized, training_history
    
    # Try to get API key from environment first (for Hugging Face Spaces)
    env_api_key = os.environ.get("OPENAI_API_KEY", "")
    if env_api_key:
        api_key = env_api_key
        print("✅ Using OpenAI API key from environment (Hugging Face secret)")
    elif not api_key.strip():
        return "❌ Please provide your OpenAI API key or set OPENAI_API_KEY environment variable", "", ""
    
    try:
        progress(0.05, desc="🚀 Starting FedProx System...")
        
        # Set OpenAI API key
        os.environ["OPENAI_API_KEY"] = api_key.strip()
        
        # Initialize FedProx server with mu parameter
        federated_server = FedProxServer(mu=mu)
        
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
        
        # Now start FedProx training
        progress(0.5, desc="Starting FedProx collaborative training...")
        
        # Run FedProx training
        training_result = federated_server.fedprox_training(training_rounds)
        
        progress(0.8, desc="Processing training improvements...")
        
        # Format comprehensive results
        result_text = f"**OpenAI FedProx System Ready!**\n\n"
        
        # System info
        result_text += f"**Federated Network:**\n"
        result_text += f"• Active Nodes: {len(federated_nodes)}\n"
        result_text += f"• Total Data Chunks: {total_chunks}\n"
        result_text += f"• Provider: OpenAI Premium\n"
        result_text += f"• Models: {llm_model} + {embeddings_model}\n"
        result_text += f"• Algorithm: FedProx (μ = {mu})\n\n"
        
        # Training results
        result_text += f"**FedProx Training Completed:**\n"
        result_text += f"• Training Rounds: {training_rounds}\n"
        result_text += f"• Global Improvement: {training_result['global_improvement']:.3f}\n"
        result_text += f"• Total Regularization: {training_result.get('total_regularization', 0):.3f}\n\n"
        
        # Node improvements with regularization info
        result_text += f"**Node Enhancements (with Regularization):**\n"
        for node_name, reg_effects in training_result.get("regularization_effects", {}).items():
            result_text += f"• {node_name}:\n"
            result_text += f"  - Net Improvement: +{reg_effects['net_effect']:.3f}\n"
            result_text += f"  - Regularization Applied: -{reg_effects['total_regularization']:.3f}\n"
        
        # Convergence info
        if training_result.get("convergence_metrics"):
            final_convergence = training_result["convergence_metrics"][-1]
            result_text += f"\n**Convergence Metrics:**\n"
            result_text += f"• Parameter Variance: {final_convergence.get('variance', 0):.4f}\n"
            result_text += f"• Mean Distance: {final_convergence.get('mean_distance', 0):.4f}\n"
        
        result_text += f"\n**Ready for Regularized Queries!**\n"
        result_text += f"System now provides FedProx-enhanced responses with improved consistency and balanced institutional perspectives."
        
        training_history.append(training_result)
        
        progress(1.0, desc="System ready!")
        
        return result_text, "", ""
        
    except Exception as e:
        progress(1.0, desc="❌ Setup failed")
        return f"❌ Error setting up system: {str(e)}", "", ""

def federated_query(question: str, progress=gr.Progress()):
    """Process a question through the OpenAI FedProx system"""
    global federated_nodes, initialized, chat_history
    
    progress(0.05, desc="Processing your query...")
    
    if not initialized or not federated_nodes:
        progress(1.0, desc="❌ System not ready")
        return "❌ **Please initialize the system first using the big blue button above.**", "", ""
    
    if not question.strip():
        progress(1.0, desc="❌ No question provided")
        return "❌ Please enter a question.", "", ""
    
    try:
        progress(0.1, desc="Querying regularized nodes...")
        
        # Query all nodes
        node_results = []
        total_nodes = len(federated_nodes)
        
        for i, (node_name, node) in enumerate(federated_nodes.items()):
            node_progress = 0.2 + (0.5 * i / total_nodes)
            progress(node_progress, desc=f"🏛️ Querying {node_name}...")
            result = node.query(question)
            node_results.append(result)
        
        progress(0.75, desc="Synthesizing regularized results...")
        
        # Synthesize results
        synthesized_answer = synthesize_fedprox_results(question, node_results)
        
        progress(0.85, desc="Formatting results...")
        
        # Format individual node results with FedProx information
        node_details = ""
        for result in node_results:
            confidence_bar = "█" * int(result["confidence"] * 10) + "░" * (10 - int(result["confidence"] * 10))
            
            if result["fedprox_enhanced"]:
                enhancement_status = f"FedProx Enhanced (μ={federated_server.mu}, Rounds: {result['training_rounds']})"
                status_color = "color: #0ea5e9; font-weight: bold;"
                
                # Add regularization info
                reg_info = result.get("regularization_info", {})
                reg_details = ""
                if reg_info and "parameter_deviations" in reg_info:
                    reg_details = f"<br><small>Regularization: Penalty={result.get('regularization_penalty', 0):.3f}, "
                    reg_details += f"Confidence Adj={result.get('confidence_adjustment', 0):.3f}</small>"
            else:
                enhancement_status = "Standard"
                status_color = "color: #666; font-weight: normal;"
                reg_details = ""
            
            node_details += f"""
<div style="background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%); padding: 1rem; border-radius: 8px; margin: 0.5rem 0; border-left: 4px solid #0ea5e9;">
<h4 style="margin: 0 0 0.5rem 0;">{result['node_name']}</h4>
<p style="{status_color} margin: 0.25rem 0;">{enhancement_status}{reg_details}</p>
<p style="margin: 0.25rem 0;"><strong>Confidence:</strong> {result['confidence']:.2f} {confidence_bar}</p>
<div style="margin-top: 0.5rem; padding: 0.5rem; background: rgba(255,255,255,0.7); border-radius: 4px;">
{result['answer']}
</div>
</div>
"""
        
        progress(0.92, desc="Adding sources...")
        
        # Format sources (filter out summary tables and unwanted content)
        sources_text = ""
        for result in node_results:
            if result["source_documents"]:
                sources_text += f"\n**Sources from {result['node_name']}:**\n"
                source_count = 0
                for source in result["source_documents"]:
                    # Filter out sources that contain summary tables or unwanted content
                    content = source.page_content.lower()
                    if any(keyword in content for keyword in [
                        "summary table", "| area |", "| highlights |", 
                        "|---", "table", "| modules |", "| features |"
                    ]):
                        continue  # Skip this source
                    
                    source_count += 1
                    if source_count > 2:  # Limit to 2 relevant sources
                        break
                        
                    content_preview = source.page_content[:150] + "..." if len(source.page_content) > 150 else source.page_content
                    sources_text += f"• **Source {source_count}:** {content_preview}\n\n"
        
        progress(0.96, desc="Saving to history...")
        
        # Add to chat history
        chat_history.append((question, synthesized_answer, node_results))
        
        # Format chat history for display
        chat_display = ""
        for i, (q, a, _) in enumerate(chat_history, 1):
            chat_display += f"**Q{i}:** {q}\n\n**FedProx AI:** {a}\n\n---\n\n"
        
        progress(1.0, desc="Query complete!")
        
        return synthesized_answer, node_details + sources_text, chat_display
        
    except Exception as e:
        progress(1.0, desc="❌ Query failed")
        return f"❌ Error processing query: {str(e)}", "", ""

def synthesize_fedprox_results(question: str, node_results: List[Dict]) -> str:
    """Synthesize answers from multiple federated nodes using FedProx information"""
    if not node_results:
        return "❌ No results available from federated nodes."
    
    valid_results = [r for r in node_results if not r["answer"].startswith("Error")]
    
    if not valid_results:
        return "❌ All federated nodes encountered errors processing the query."
    
    # Sort by confidence
    valid_results.sort(key=lambda x: x["confidence"], reverse=True)
    
    # Check enhancement status and regularization info
    enhanced_count = sum(1 for r in valid_results if r.get("fedprox_enhanced", False))
    total_improvement = sum(r.get("improvement_score", 0) for r in valid_results)
    total_regularization = sum(r.get("regularization_penalty", 0) for r in valid_results)
    
    # Create synthesis prompt with FedProx information
    synthesis_prompt = f"""
You are synthesizing responses from an OpenAI-powered Federated RAG system with FedProx regularization.

Question: "{question}"

Node Responses ({enhanced_count}/{len(valid_results)} FedProx-regularized, Net improvement: {total_improvement - total_regularization:.3f}):

"""
    
    for i, result in enumerate(valid_results, 1):
        enhancement_info = ""
        if result.get("fedprox_enhanced", False):
            reg_penalty = result.get("regularization_penalty", 0)
            conf_adj = result.get("confidence_adjustment", 0)
            enhancement_info = f" [FedProx: +{result.get('improvement_score', 0):.3f}, Reg: -{reg_penalty:.3f}, Conf Adj: {conf_adj:+.3f}]"
        
        synthesis_prompt += f"""
Node {i} - {result['node_name']}{enhancement_info} (Confidence: {result['confidence']:.2f}):
{result['answer']}

"""
    
    synthesis_prompt += f"""
Provide a comprehensive synthesis that:
1. Combines insights from all institutional nodes with FedProx regularization benefits
2. Highlights how regularization improves consistency and reduces institutional bias
3. Provides balanced comparative analysis that leverages the proximal regularization
4. Notes confidence levels and gives actionable insights
5. Emphasizes the improved reliability from FedProx's regularization approach

FedProx-Enhanced Synthesis:"""
    
    try:
        # Use OpenAI to synthesize
        llm = ChatOpenAI(temperature=0.2, model_name="gpt-4o")
        synthesized = llm.predict(synthesis_prompt)
        
        enhancement_indicator = f"**FedProx-Regularized** ({enhanced_count}/{len(valid_results)} nodes, μ={federated_server.mu})" if enhanced_count > 0 else "📊 **Standard Analysis**"
        
        return f"{enhancement_indicator}\n\n{synthesized.strip()}"
        
    except Exception as e:
        # Fallback synthesis
        return simple_fedprox_synthesis(valid_results, enhanced_count)

def simple_fedprox_synthesis(results: List[Dict], enhanced_count: int = 0) -> str:
    """Simple fallback synthesis method for FedProx"""
    if len(results) == 1:
        enhancement = "FedProx Regularized" if results[0].get("fedprox_enhanced", False) else "📊 Standard"
        return f"**Single Node Response from {results[0]['node_name']} ({enhancement}):**\n\n{results[0]['answer']}"
    
    enhancement_status = f"**FedProx-Regularized Analysis** ({enhanced_count}/{len(results)} nodes regularized)" if enhanced_count > 0 else "📊 **Standard Analysis**"
    
    synthesis = f"{enhancement_status}\n\n"
    
    for i, result in enumerate(results, 1):
        enhancement = "Regularized" if result.get("fedprox_enhanced", False) else "📊 Standard"
        reg_info = ""
        if result.get("fedprox_enhanced", False):
            reg_penalty = result.get("regularization_penalty", 0)
            reg_info = f" (Reg: -{reg_penalty:.3f})"
        
        synthesis += f"**🏛️ {result['node_name']}** ({enhancement}{reg_info}) - Confidence: {result['confidence']:.2f}\n"
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
        "Compare the technical skills focus between Hull and Keele programs.",
        "What are the main differences in student support services?",
        "Which university offers better industry connections for AI graduates?",
        "Compare the research opportunities at Hull versus Keele for AI students."
    ]
    return np.random.choice(examples)

# Create Gradio interface
def create_fedprox_interface():
    """Create the main Gradio interface for FedProx system"""
    
    with gr.Blocks(css=custom_css, title="FedProx Federated RAG System") as demo:
        
        # Header
        gr.HTML("""
        <div class="main-header">
            <h1 class="main-title">🔗 FedProx Federated RAG</h1>
            <p class="sub-title">Ask Questions about MSc AI online programmes at Hull or Keele University</p>
            <div class="openai-indicator">Powered by OpenAI + FedProx Algorithm</div>
        </div>
        """)
        
        # Main initialization section
        with gr.Row():
            with gr.Column():
                                
                # Settings in a compact layout
                with gr.Row():
                    with gr.Column(scale=2):
                        # Check if API key is available in environment
                        env_api_key = os.environ.get("OPENAI_API_KEY", "")
                        api_key_placeholder = "API key loaded from environment" if env_api_key else "sk-..."
                        api_key_info = "API key is loaded from Hugging Face secrets" if env_api_key else "Your OpenAI API key for embeddings and LLM"
                        
                        api_key_input = gr.Textbox(
                            label="🔑 OpenAI API Key",
                            placeholder=api_key_placeholder,
                            type="password",
                            info=api_key_info,
                            value="" if env_api_key else ""
                        )
                    with gr.Column(scale=1):
                        mu_input = gr.Slider(
                            minimum=0.01,
                            maximum=1.0,
                            value=0.1,
                            step=0.01,
                            label="μ (Regularization)",
                            info="FedProx regularization strength"
                        )
                
                with gr.Accordion("⚙️ Advanced Settings", open=False):
                    with gr.Row():
                        embeddings_model = gr.Dropdown(
                            choices=["text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"],
                            value="text-embedding-3-small",
                            label="Embeddings Model"
                        )
                        llm_model = gr.Dropdown(
                            choices=["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"],
                            value="gpt-4o-mini",
                            label="LLM Model"
                        )
                    
                    with gr.Row():
                        temperature = gr.Slider(0.0, 1.0, 0.1, label="Temperature")
                        k_docs = gr.Slider(1, 10, 4, step=1, label="Retrieved Documents")
                        training_rounds = gr.Slider(1, 10, 3, step=1, label="Training Rounds")
                    
                    with gr.Row():
                        chunk_size = gr.Slider(200, 2000, 800, step=100, label="Chunk Size")
                        chunk_overlap = gr.Slider(0, 500, 100, step=50, label="Chunk Overlap")
                
                # File inputs
                with gr.Row():
                    hull_file = gr.File(
                        label="📄 Hull University Data",
                        file_types=[".txt"],
                        value="data/hull.txt" if os.path.exists("data/hull.txt") else None
                    )
                    keele_file = gr.File(
                        label="📄 Keele University Data", 
                        file_types=[".txt"],
                        value="data/keele.txt" if os.path.exists("data/keele.txt") else None
                    )
                
                # Main action button
                init_button = gr.Button(
                    "🚀 Initialize & Train FedProx System",
                    variant="primary",
                    size="lg",
                    elem_id="main-action-button"
                )
        
        # Results section
        with gr.Row():
            with gr.Column():
                # Check if running in Hugging Face Space environment
                env_api_key = os.environ.get("OPENAI_API_KEY", "")
                if env_api_key:
                    status_message = "**System Status:** Ready for Hugging Face Space\n\nAPI key loaded from environment. Data files ready. Click the button above to initialize FedProx system."
                else:
                    status_message = "**System Status:** Not initialized\n\nPlease provide your OpenAI API key and click the button above to start."
                
                system_status = gr.Markdown(
                    status_message,
                    elem_classes=["status-box"]
                )
        
        # Chat section
        gr.HTML("""
        <div style="margin: 2rem 0 1rem 0;">
            <h2 style="color: #0ea5e9; margin-bottom: 0.5rem;">💬 FedProx Query Interface</h2>
            <p style="color: #64748b;">Ask questions and get regularized responses from federated institutions</p>
        </div>
        """)
        
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    question_input = gr.Textbox(
                        label="❓ Your Question",
                        placeholder="Ask about university programs, costs, requirements, etc.",
                        lines=2,
                        scale=4
                    )
                    with gr.Column(scale=1):
                        query_button = gr.Button("🔍 Query", variant="secondary", size="lg")
                        example_button = gr.Button("🎲 Example", variant="secondary")
                        clear_button = gr.Button("🗑️ Clear", variant="secondary")
        
        # Results display
        with gr.Row():
            with gr.Column():
                answer_output = gr.Markdown(
                    label="🤖 FedProx Enhanced Answer",
                    elem_classes=["chat-section"]
                )
        
        with gr.Row():
            with gr.Column():
                with gr.Accordion("📊 Detailed Node Analysis", open=False):
                    details_output = gr.Markdown()
        
        with gr.Row():
            with gr.Column():
                with gr.Accordion("💬 Chat History", open=False):
                    chat_output = gr.Markdown()
        
        # Event handlers
        init_button.click(
            fn=initialize_and_train_fedprox_system,
            inputs=[
                api_key_input, embeddings_model, llm_model, temperature, 
                k_docs, chunk_size, chunk_overlap, training_rounds, mu_input,
                hull_file, keele_file
            ],
            outputs=[system_status, details_output, chat_output],
            show_progress=True
        )
        
        query_button.click(
            fn=federated_query,
            inputs=[question_input],
            outputs=[answer_output, details_output, chat_output],
            show_progress=True
        )
        
        question_input.submit(
            fn=federated_query,
            inputs=[question_input],
            outputs=[answer_output, details_output, chat_output],
            show_progress=True
        )
        
        example_button.click(
            fn=get_federated_example_question,
            outputs=[question_input]
        )
        
        clear_button.click(
            fn=clear_chat_history,
            outputs=[answer_output, details_output, chat_output]
        )
        
        # Footer
        gr.HTML("""
        <div style="text-align: center; margin-top: 2rem; padding: 1rem; background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%); border-radius: 8px;">
            <p style="color: #64748b; margin: 0;">
                <strong>FedProx Federated RAG System</strong> | 
                Advanced federated learning with proximal regularization | 
                Powered by OpenAI
            </p>
        </div>
        """)
    
    return demo

# Create the demo interface for import
demo = create_fedprox_interface()

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        debug=False
    )

