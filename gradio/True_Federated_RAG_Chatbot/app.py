#!/usr/bin/env python3
"""
Fully Separated Federated RAG Chatbot
Main entry point for the application
"""

from True_Federated_RAG_Chatbot_g import create_interface

if __name__ == "__main__":
    # Create and launch the improved Gradio interface
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )

