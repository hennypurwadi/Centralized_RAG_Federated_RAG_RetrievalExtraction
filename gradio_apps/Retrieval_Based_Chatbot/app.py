import gradio as gr
import os
import pickle
import random
from sentence_transformers import SentenceTransformer, util

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

# Initialize model and data (load once at startup)
try:
    # Load the saved embeddings
    embedding_file_path = './data/MiniLM_embeddings.pkl'
    answer_embeddings = load_embeddings(embedding_file_path)
    
    # Load the ground truth answers from the file
    ground_truth_answers = load_ground_truth_answers('./data/msc_ai_hullonline_short.txt')
    
    # Load the embedding model
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    data_loaded = True
except Exception as e:
    data_loaded = False
    error_message = f"Error loading data: {str(e)}"

# Function to get a random example question
def get_random_example_question():
    """
    Returns a random example question from the predefined list
    """
    example_questions = [
        "What will be on my degree certificate after completing an online Masters course at the University of Hull?",
        "Is there a graduation ceremony for online students at the University of Hull?",
        "What is the phone number to reach the course advisers at the University of Hull?",
        "What is the email address for enquiries about online courses at the University of Hull?",
        "What are the payment options available for paying tuition fees at the University of Hull Online?",
        "Is there any discount available for referring a friend to the University of Hull Online?"
    ]
    return random.choice(example_questions)

def get_answer(question):
    """
    Process user question and return the most relevant answer
    """
    if not data_loaded:
        return f"❌ {error_message}"
    
    if not question or question.strip() == "":
        return "⚠️ Please enter a question."
    
    try:
        # Generate embedding for the user's question
        question_embedding = model.encode([question], convert_to_tensor=True)
        
        # Find the most relevant answer
        most_similar_index = perform_embeddings(question_embedding, answer_embeddings)
        
        # Format the response
        answer = ground_truth_answers[most_similar_index]
        
        return f"**Q:** {question}\n\n**A:** {answer}"
        
    except Exception as e:
        return f"❌ An error occurred while processing your question: {str(e)}"

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

.gr-button-secondary, .gr-button[data-testid="secondary-button"] {
    background: linear-gradient(135deg, #fce7f3 0%, #fbcfe8 100%) !important;
    border: 2px solid #f472b6 !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 12px 24px !important;
    transition: all 0.3s ease !important;
    color: #be185d !important;
}

.gr-button-secondary:hover, .gr-button[data-testid="secondary-button"]:hover {
    background: linear-gradient(135deg, #f472b6 0%, #ec4899 100%) !important;
    border-color: #ec4899 !important;
    color: white !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(244, 114, 182, 0.3) !important;
}

/* Additional targeting for Gradio button variants */
button[variant="secondary"] {
    background: linear-gradient(135deg, #fce7f3 0%, #fbcfe8 100%) !important;
    border: 2px solid #f472b6 !important;
    color: #be185d !important;
}

button[variant="secondary"]:hover {
    background: linear-gradient(135deg, #f472b6 0%, #ec4899 100%) !important;
    color: white !important;
}
"""

# Create the Gradio interface
with gr.Blocks(css=custom_css, title="Retrieval-Based Chatbot", theme=gr.themes.Soft()) as demo:
    # Header section
    gr.HTML("""
        <div class="main-header">
            <h1 class="main-title">Retrieval-Based Methods with Embeddings</h1>
            <h1 class="main-title">(Without Generation)</h1>
            <p class="sub-title">about the MSc AI Online Program at the University of Hull</p>
        </div>
    """)
    
    # Description section
    gr.HTML("""
        <div class="description">
            <p>Please ask questions about the MSc Artificial Intelligence Online program at the University of Hull. 
            This chatbot uses sentence embeddings to retrieve the most relevant answers from a knowledge base.</p>
        </div>
    """)
    
    # Main interface
    with gr.Group(elem_classes=["input-section"]):
        question_input = gr.Textbox(
            label="Enter your question:",
            placeholder="e.g., What are the admission requirements for the MSc AI program?",
            lines=2,
            show_label=True
        )
        
        # Buttons row
        with gr.Row():
            submit_btn = gr.Button(
                "Get Answer", 
                variant="primary", 
                size="lg",
                scale=2
            )
            example_btn = gr.Button(
                "Example Questions", 
                variant="secondary", 
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
            <p> <strong>How it works:</strong> This system uses sentence transformers to encode your question and find the most semantically similar answer from the knowledge base.</p>
            <p> <strong>About:</strong> MSc Artificial Intelligence Online Program - University of Hull</p>
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
        fn=get_random_example_question,
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

