import gradio as gr
import os
import pickle
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
import random

# Function to load the saved embeddings
def load_embeddings(embedding_file_path):
    with open(embedding_file_path, 'rb') as f:
        return pickle.load(f)

# Function to load the ground truth answers from a file
def load_ground_truth_answers(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        answers = file.readlines()
    return [answer.strip() for answer in answers]

# Function to generate embeddings and find the most relevant answers
def perform_embeddings(question_embedding, answer_embeddings):
    # Compute cosine similarity between the question and answers
    similarities = util.pytorch_cos_sim(question_embedding, answer_embeddings)
    
    # Find the index of the most similar answer
    most_similar_index = similarities.argmax()
    
    # Retrieve the most relevant answer
    return most_similar_index

# Initialize models and data (load once at startup)
try:
    # Load the embedding model for retrieval
    retrieval_model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Load DistilBERT for extractive QA (NOT GENERATION)
    qa_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-distilled-squad")
    qa_model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-uncased-distilled-squad")
    qa_pipeline = pipeline("question-answering", model=qa_model, tokenizer=qa_tokenizer)
    
    # Load Hull data
    hull_embeddings = load_embeddings('./data/MiniLM_embeddings_hull.pkl')
    hull_ground_truth = load_ground_truth_answers('./data/msc_ai_hullonline_short.txt')
    
    # Load Keele data
    keele_embeddings = load_embeddings('./data/MiniLM_embeddings_keele.pkl')
    keele_ground_truth = load_ground_truth_answers('./data/msc_ai_keeleonline_short.txt') 
    
    data_loaded = True
except Exception as e:
    data_loaded = False
    error_message = f"Error loading data: {str(e)}"

def get_answer(question):
    """
    Process user question and return the most relevant answer using retrieval and extractive QA.
    """
    if not data_loaded:
        return f" {error_message}"
    
    if not question or question.strip() == "":
        return " Please enter a question."
    
    # Determine which university's data to use
    question_lower = question.lower()
    if "hull" in question_lower:
        current_embeddings = hull_embeddings
        current_ground_truth = hull_ground_truth
        university_name = "University of Hull"
    elif "keele" in question_lower:
        current_embeddings = keele_embeddings
        current_ground_truth = keele_ground_truth
        university_name = "Keele University"
    else:
        return "Please specify either **Hull** or **Keele** in your question."

    try:
        # Retrieval: Generate embedding for the user's question
        question_embedding = retrieval_model.encode([question], convert_to_tensor=True)
        
        # Find the most relevant document/answer for retrieval
        most_similar_index = perform_embeddings(question_embedding, current_embeddings)
        retrieved_document = current_ground_truth[most_similar_index]
        
        # Extractive QA: Use DistilBERT to find the answer within the retrieved document
        qa_input = {
            'question': question,
            'context': retrieved_document
        }
        extractive_answer = qa_pipeline(qa_input)['answer']
        
        return f"**Q:** {question}\n\n**A (from {university_name}):** {extractive_answer}\n\n**Retrieved Document:** {retrieved_document}"
        
    except Exception as e:
        return f" An error occurred while processing your question: {str(e)}"

# List of example questions
example_questions = [
    "How long does the MSc Artificial Intelligence course at the University of Hull take to complete?",
    "When are the start dates for the MSc Computer Science with Artificial Intelligence at Keele University?",
    "What is the phone number to reach the course advisers at the University of Hull?",
    "How long does the MSc Computer Science with Artificial Intelligence at Keele University take to complete?",
    "What is the total cost of the MSc Artificial Intelligence Online program at the University of Hull?",
    "What is the total cost of the MSc Computer Science with Artificial Intelligence at Keele University?"
]

def get_random_example():
    return random.choice(example_questions)

# Custom CSS for better styling
custom_css = """
.gradio-container {
    max-width: 900px !important;
    margin: auto !important;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

.main-header {
    text-align: center;
    margin-bottom: 2rem;
    padding: 2rem;
    background: linear-gradient(135deg, #fbcfe8 0%, #f472b6 100%);
    border-radius: 15px;
    color: white;
    box-shadow: 0 8px 32px rgba(0,0,0,0.1);
}

.main-title {
    font-size: 2.2rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
    line-height: 1.2;
}

.sub-title {
    font-size: 1.1rem;
    opacity: 0.9;
    margin-bottom: 0;
}

.description {
    font-size: 1rem;
    color: #34495e;
    margin: 2rem 0;
    padding: 1.5rem;
    background: linear-gradient(135deg, #fce7f3 0%, #fbcfe8 100%);
    border-radius: 12px;
    border-left: 5px solid #f472b6;
    box-shadow: 0 4px 16px rgba(0,0,0,0.05);
}

.input-section {
    margin-bottom: 1.5rem;
    background: linear-gradient(135deg, #fff0f5 0%, #ffe4e1 100%);
    padding: 1.5rem;
    border-radius: 12px;
    box-shadow: 0 4px 16px rgba(0,0,0,0.05);
}

.output-section {
    margin-top: 1.5rem;
    background: linear-gradient(135deg, #fff0f5 0%, #ffe4e1 100%);
    padding: 1.5rem;
    border-radius: 12px;
    box-shadow: 0 4px 16px rgba(0,0,0,0.05);
}

.footer-info {
    text-align: center;
    margin-top: 2rem;
    padding: 1.5rem;
    background: linear-gradient(135deg, #fce7f3 0%, #fbcfe8 100%);
    border-radius: 12px;
    font-size: 0.9rem;
    color: #7f8c8d;
    box-shadow: 0 4px 16px rgba(0,0,0,0.05);
}

.gr-button-primary {
    background: linear-gradient(135deg, #f472b6 0%, #ec4899 100%) !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 12px 24px !important;
    transition: all 0.3s ease !important;
}

.gr-button-primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(244, 114, 182, 0.4) !important;
}

.gr-textbox {
    border-radius: 8px !important;
    border: 2px solid #e9ecef !important;
    transition: all 0.3s ease !important;
}

.gr-textbox:focus {
    border-color: #f472b6 !important;
    box-shadow: 0 0 0 3px rgba(244, 114, 182, 0.1) !important;
}

.gr-button-secondary {
    background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%) !important;
    border: 2px solid #dee2e6 !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    padding: 8px 16px !important;
    transition: all 0.3s ease !important;
    color: #495057 !important;
    font-size: 0.9rem !important;
    margin: 2px !important;
}

.gr-button-secondary:hover {
    background: linear-gradient(135deg, #f472b6 0%, #ec4899 100%) !important;
    border-color: #f472b6 !important;
    color: white !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(244, 114, 182, 0.3) !important;
}
"""

# Create the Gradio interface
with gr.Blocks(css=custom_css, title="Retrieval-Based Chatbot", theme=gr.themes.Soft()) as demo:
    # Header section
    gr.HTML("""
        <div class="main-header">
            <h1 class="main-title">Retrieval-Based Methods with Embeddings and Extractive QA</h1>
            <p class="sub-title">about the MSc AI Online Programmes at University of Hull or Keele University</p>
        </div>
    """)
    
    # Description section
    gr.HTML("""
        <div class="description">
            <p>Please ask questions about MSc Artificial Intelligence Online program at the University of Hull or Keele University. 
            This chatbot uses MiniLM embeddings for retrieval and DistilBERT for extractive answers. 
            Please include the word <b>Hull</b> or <b>Keele</b> in your question.</p>
        </div>
    """)
    
    # Main interface
    with gr.Group(elem_classes=["input-section"]):
        question_input = gr.Textbox(
            label="Enter your question:",
            placeholder="e.g., What are the admission requirements for the MSc AI program at Hull?",
            lines=2,
            show_label=True
        )
        
        # Example questions section
        gr.HTML("<h3 style='margin-top: 1rem; margin-bottom: 0.5rem; color: #f472b6;'> Try these example questions:</h3>")
        
        with gr.Row():
            example_btn = gr.Button(
                "Examples",
                size="sm",
                variant="secondary"
            )
            submit_btn = gr.Button(
                "Get Answer", 
                variant="primary", 
                size="lg",
                scale=1
            )
    
    # Output section
    with gr.Group(elem_classes=["output-section"]):
        answer_output = gr.Markdown(
            label="Answer:",
            show_label=True,
            visible=True
        )
    
    # Footer information
    gr.HTML("""
        <div class="footer-info">
            <p> <strong>How it works:</strong> This system uses MiniLM sentence embeddings for retrieval and DistilBERT for extractive answers.</p>
            <p> <strong>About:</strong> MSc Artificial Intelligence Online Programmes - University of Hull & Keele University</p>
        </div>
    """)
    
    # Event handlers
    submit_btn.click(
        fn=get_answer,
        inputs=[question_input],
        outputs=[answer_output]
    )
    
    question_input.submit(
        fn=get_answer,
        inputs=[question_input],
        outputs=[answer_output]
    )
    
    # Example button event handler
    example_btn.click(
        fn=get_random_example,
        outputs=[question_input]
    )

# Launch the app
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )


