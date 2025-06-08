
import streamlit as st
import os
import pickle
from sentence_transformers import SentenceTransformer, util
import time  # For simulating progress

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

# Streamlit app
def main():
    # Displaying the title in two rows using markdown
    st.markdown("<h1 style='text-align: center;'>Retrieval-Based Methods with Embeddings (Without Generation) </h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center;'>about the MSc AI Online Program at the University of Hull</h2>", unsafe_allow_html=True)

    # Placeholder for the introduction or context
    st.write("""
    Please ask questions about the MSc Artificial Intelligence Online program at the University of Hull.
    """)

    # Input form
    user_question = st.text_input("Enter your question:")

    if st.button("Get Answer"):
        if user_question:
            with st.spinner('Processing...'):
                # Initialize progress bar
                progress_bar = st.progress(0)

                # Simulate loading and processing steps with progress updates
                for percent_complete in range(1, 101):
                    time.sleep(0.02)  # Simulate some processing delay
                    progress_bar.progress(percent_complete)

                # Load the saved embeddings
                embedding_file_path = './data/MiniLM_embeddings.pkl'
                answer_embeddings = load_embeddings(embedding_file_path)

                # Load the ground truth answers from the file
                ground_truth_answers = load_ground_truth_answers('./data/msc_ai_hullonline_short.txt')

                # Load the embedding model
                model = SentenceTransformer("all-MiniLM-L6-v2")

                # Generate embedding for the user's question
                question_embedding = model.encode([user_question], convert_to_tensor=True)

                # Find the most relevant answer
                most_similar_index = perform_embeddings(question_embedding, answer_embeddings)

                # Display the question and retrieved answer
                st.write(f"**Q: {user_question}**")
                st.write(f"**A: {ground_truth_answers[most_similar_index]}**")
        else:
            st.warning("Please enter a question.")

if __name__ == "__main__":
    main()
