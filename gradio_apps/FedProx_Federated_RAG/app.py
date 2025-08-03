#!/usr/bin/env python3
"""
Fully Separated Federated RAG Chatbot
Main entry point for the application
"""

from FedProx_Federated_RAG import demo

if __name__ == "__main__":
    # Launch the Federated RAG Gradio interface
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True)