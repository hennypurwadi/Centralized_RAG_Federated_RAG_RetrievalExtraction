# 🔗 Federated RAG Chatbot

A privacy-preserving Retrieval-Augmented Generation (RAG) system that compares AI Master's programs from Hull University and Keele University while maintaining complete institutional data privacy.

## 🌟 Features

- **Federated Architecture**: Each institution's data is processed in separate, isolated nodes
- **Privacy-Preserving**: Only synthesized results are shared, maintaining complete data privacy
- **Comparative Analysis**: Enables comprehensive program comparisons without exposing raw data
- **Beautiful Interface**: Modern Gradio interface with custom styling
- **Real-time Processing**: Query multiple nodes simultaneously for comprehensive answers

## 🏛️ Institutions Covered

- **University of Hull**: MSc Artificial Intelligence Online
- **Keele University**: MSc Computer Science with Artificial Intelligence

## 🚀 How It Works

1. **Federated Nodes**: Each university's data is processed in separate RAG nodes
2. **Independent Processing**: Queries are sent to all nodes independently
3. **Result Synthesis**: Responses are synthesized without exposing raw institutional data
4. **Privacy Maintained**: Complete data privacy while enabling informed comparisons

## 📋 Requirements

- OpenAI API Key (for embeddings and LLM processing)
- Python 3.11+
- Dependencies listed in `requirements.txt`

## 🔧 Setup Instructions

### For HuggingFace Spaces:

1. Create a new Space on HuggingFace
2. Upload all files from this repository
3. Set the Space SDK to "Gradio"
4. The app will automatically start with `app.py`

### For Local Development:

```bash
# Clone or download the files
cd federated_rag_app

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

## 📁 File Structure

```
federated_rag_app/
├── app.py                              # Main entry point
├── True_Federated_RAG_Chatbot_g.py    # Core federated RAG implementation
├── requirements.txt                    # Python dependencies
├── msc_ai_hullonline_short.txt        # Hull University data
├── msc_ai_keeleonline.txt             # Keele University data
└── README.md                          # This file
```

## 🎯 Usage

1. **Initialize System**: Enter your OpenAI API key and click "Initialize Federated System"
2. **Ask Questions**: Enter comparative questions about the AI programs
3. **View Results**: Get synthesized answers from both institutions
4. **Explore Details**: Check individual node results and confidence scores

## 💡 Sample Questions

- Compare the total program costs between MSc Artificial Intelligence at University of Hull online and Keele University
- What are the entry requirement differences between MSc Artificial Intelligence at University of Hull online and Keele University?
- Which program between MSc Artificial Intelligence at University of Hull online and Keele University offers more flexibility for working professionals?
- Compare the technical skills and programming languages covered between MSc Artificial Intelligence at University of Hull online and Keele University
- How do the start dates and program durations differ between MSc Artificial Intelligence at University of Hull online and Keele University?
- What are the assessment method differences between MSc Artificial Intelligence at University of Hull online and Keele University?
- Which program is better suited for career changers between MSc Artificial Intelligence at University of Hull online and Keele University?

## 🔒 Privacy Features

- **Data Isolation**: Each institution's data remains in separate nodes
- **No Raw Data Sharing**: Only processed results are combined
- **Institutional Privacy**: Raw documents never leave their respective nodes
- **Secure Processing**: All processing happens within isolated environments

## 🛠️ Technical Details

- **Framework**: Gradio for the web interface
- **RAG Implementation**: LangChain with FAISS vector stores
- **Embeddings**: OpenAI text-embedding-3-small/large
- **LLM**: OpenAI GPT-4o or GPT-3.5-turbo
- **Architecture**: Federated learning approach with independent nodes

## 📊 Configuration Options

- **Model Settings**: Choose between different OpenAI models
- **Retrieval Settings**: Adjust number of documents retrieved per node
- **Processing Settings**: Customize chunk size and overlap for document processing
- **Data Sources**: Configure paths to institutional data files

## 🤝 Contributing

This is a demonstration of federated RAG architecture. In a real-world scenario:
- Each institution would run their own node
- Only synthesized results would be shared
- Complete data sovereignty would be maintained

## 📄 License

This project is for educational and demonstration purposes.

## 🔗 Links

- [University of Hull MSc AI Online](https://www.hull.ac.uk/)
- [Keele University MSc Computer Science with AI](https://www.keele.ac.uk/)
- [OpenAI API](https://openai.com/api/)
- [LangChain](https://langchain.com/)
- [Gradio](https://gradio.app/)

---

**Note**: This is a demonstration of Federated RAG architecture. In production, each institution would maintain their own node for complete data privacy.

