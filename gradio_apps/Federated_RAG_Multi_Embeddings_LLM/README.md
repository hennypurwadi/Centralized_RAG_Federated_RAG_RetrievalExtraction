---
title: Federated_RAG_Multi_Embeddings_LLM
emoji: 📊
colorFrom: purple
colorTo: purple
sdk: gradio
sdk_version: 5.36.2
app_file: app.py
pinned: false
license: apache-2.0
short_description: 'Ask questions about online MSc AI at Uni of Hull and Keele '
---

Providers Option:

Option 1: Free Huggingface Models (No API Key Required)

Option 2: OpenAI Models (API Key Required)


Available Models

1.Huggingface (FREE)

Embeddings: 
sentence-transformers/all-MiniLM-L6-v2 

Language Models:
microsoft/DialoGPT-medium 
distilgpt2

2.OpenAI (Paid) Embeddings:
text-embedding-3-small
text-embedding-3-large
text-embedding-ada-002

Language Models:
gpt-4o
gpt-4o-mini
gpt-3.5-turbo


Sample Questions

1."Compare the total program costs between MSc Artificial Intelligence at University of Hull online and Keele University online."

2."What are the entry requirement differences between MSc Artificial Intelligence at University of Hull online and Keele University?"

3."Which program offers more flexibility for working professionals?"

4."Compare the technical skills and programming languages covered between both programs."

5."How do the start dates and program durations differ?"


Technical Architecture Federated Learning Approach

Node Isolation: Each university's data processed in separate RAG nodes

Privacy Preservation: No raw data sharing between institutions


Synthesis Layer: 

Combines insights while maintaining privacy

Multi-Provider Support

Abstraction Layer: Unified interface for different AI providers

Dynamic Switching: Runtime provider and model selection

Fallback Mechanisms: Graceful handling of model failures


Technology Stack

Frontend: Gradio with custom CSS styling

Backend: LangChain for RAG implementation

Embeddings: OpenAI Embeddings / Huggingface Sentence Transformers

LLMs: OpenAI GPT / Huggingface Transformers

Vector Store: FAISS for efficient similarity search

Federated RAG's Data Isolation: University data never leaves its designated node