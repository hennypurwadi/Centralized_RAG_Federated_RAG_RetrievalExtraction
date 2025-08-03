---
title: FedAVG_Federated_RAG
emoji: 📊
colorFrom: purple
colorTo: red
sdk: gradio
sdk_version: 5.38.2
app_file: app.py
pinned: false
license: apache-2.0
short_description: 'Ask questions about online MSc AI at Uni of Hull and Keele '
---
# FedAvg Federated RAG Chatbot

A revolutionary chatbot that combines **Federated Averaging (FedAvg)** algorithm with **Retrieval-Augmented Generation (RAG)** to enable privacy-preserving collaborative learning across multiple institutions.

## Key Features

### Federated Averaging Integration
- **Collaborative Learning**: Multiple institutions train embedding models together without sharing raw data
- **Privacy Preservation**: Each node keeps its data local while contributing to global model improvement
- **Weight Aggregation**: Uses the FedAvg algorithm to combine model weights from different nodes
- **Enhanced Embeddings**: Collaboratively trained embeddings improve retrieval quality

### Multi-Institutional RAG
- **Separate Nodes**: Each institution maintains its own RAG system
- **Independent Querying**: Queries are processed at each node independently
- **Intelligent Synthesis**: Results are combined using advanced synthesis techniques
- **Confidence Scoring**: Each response includes confidence metrics

### Advanced Interface
- **Real-time Training**: Monitor FedAvg training progress in real-time
- **Training Metrics**: Detailed statistics for each training round
- **Enhanced Responses**: Clear indicators for FedAvg-enhanced vs standard responses
- **Interactive Controls**: Configurable training parameters and model settings

### Prerequisites
```bash
pip install -r requirements_fedavg.txt
```

### Data Setup
1. Create a `data` directory
2. Add your institution data files:
   - `hull.txt` - University of Hull program information
   - `keele.txt` - Keele University program information

### Running the Application
```bash
python Federated_RAG_Chatbot_FedAvg.py
```

## Configuration

### Provider Options
- **OpenAI**: High-quality embeddings and LLM (requires API key)
- **Huggingface**: Free models with good performance

### Training Parameters
- **Training Rounds**: Number of FedAvg aggregation rounds (1-10)
- **Local Epochs**: Training epochs per node per round (1-5)
- **Learning Rate**: Optimization learning rate (0.0001-0.01)

### Model Settings
- **Embeddings Model**: Choose embedding model based on provider
- **LLM Model**: Language model for response generation
- **Temperature**: Controls response creativity (0.0-1.0)

## How FedAvg Works

### 1. Initialization
```python
# Each node initializes its local embedding model
node.initialize_local_model()

# Server initializes global model
federated_server.initialize_global_model()
```

### 2. Training Round
```python
# Distribute global weights to all nodes
for node in nodes:
    node.set_model_weights(global_weights)

# Local training on each node
for node in nodes:
    node.local_training(epochs, learning_rate)

# Aggregate weights using FedAvg
aggregated_weights = server.aggregate_weights(node_weights, data_sizes)
```

### 3. Enhanced Querying
```python
# Query with FedAvg-enhanced embeddings
result = node.query(question)
# result includes 'fedavg_enhanced' indicator
```

## Training Process

### Local Training
Each node trains on its private data using:
- **Contrastive Learning**: Improves embedding quality
- **Local Optimization**: Adam optimizer with configurable learning rate
- **Batch Processing**: Efficient training on document chunks

### Weight Aggregation
The server combines weights using FedAvg formula:
```
w_global = Σ(n_k/n_total * w_k)
```
Where:
- `w_k` = weights from node k
- `n_k` = data size at node k
- `n_total` = total data across all nodes

### Global Model Update
- Aggregated weights update the global model
- Global model is redistributed to all nodes
- Process repeats for specified rounds

## Interface Features

### Training Section
- **Initialize System**: Set up federated nodes and global model
- **Configure Training**: Adjust rounds, epochs, and learning rate
- **Start Training**: Execute FedAvg training with progress tracking
- **View Results**: Detailed metrics for each training round

### Chat Interface
- **Enhanced Queries**: Ask questions to FedAvg-enhanced system
- **Node Results**: See individual responses from each institution
- **Synthesis**: Combined analysis leveraging all nodes
- **History**: Track conversation and training history

### Statistics Dashboard
- **Training Metrics**: Rounds completed, loss progression
- **Query Statistics**: Total queries, confidence scores
- **Enhancement Status**: FedAvg vs standard response counts
- **Node Information**: Active institutions and data sizes

## Example Usage

### 1. Initialize the System
1. Select provider (OpenAI/Huggingface)
2. Enter API key if using OpenAI
3. Configure model settings
4. Set data file paths
5. Click "Initialize FedAvg System"

### 2. Train the Models
1. Set training parameters
2. Click "Start FedAvg Training"
3. Monitor progress and results
4. Review training metrics

### 3. Query the System
1. Enter comparative question
2. Click "Ask FedAvg"
3. Review enhanced synthesis
4. Check individual node results

## Benefits of FedAvg Integration

### For Institutions
- **Privacy**: Data never leaves institutional boundaries
- **Collaboration**: Benefit from collective knowledge
- **Quality**: Improved embeddings through collaborative training
- **Control**: Maintain full control over local data and models

### For Users
- **Better Results**: Enhanced embeddings improve retrieval quality
- **Transparency**: Clear indicators of enhancement status
- **Comprehensive**: Responses leverage multiple institutional perspectives
- **Reliable**: Confidence scores help assess response quality

## Technical Architecture

### Components
1. **FedAvgServer**: Coordinates training and aggregation
2. **FederatedRAGNode**: Individual institutional nodes
3. **FedAvgEmbeddingModel**: Custom embedding model for training
4. **Gradio Interface**: Interactive web interface

### Data Flow
1. **Initialization**: Set up nodes and global model
2. **Training**: Iterative local training and global aggregation
3. **Querying**: Enhanced retrieval using trained embeddings
4. **Synthesis**: Intelligent combination of node responses

## Privacy Guarantees

- **Local Data**: All raw data remains on institutional servers
- **Model Weights Only**: Only aggregated model parameters are shared
- **Differential Privacy**: Can be extended with formal privacy guarantees
- **Secure Aggregation**: Weights can be encrypted during transmission

## Future Enhancements

- **Differential Privacy**: Add formal privacy guarantees
- **Secure Aggregation**: Encrypt weight sharing
- **Dynamic Nodes**: Support for nodes joining/leaving during training
- **Advanced Synthesis**: More sophisticated result combination methods
- **Performance Metrics**: Detailed evaluation of FedAvg benefits


Sample Questions

1."Compare the total program costs between MSc Artificial Intelligence at University of Hull online and Keele University online."

2."What are the entry requirement differences between MSc Artificial Intelligence at University of Hull online and Keele University?"

3."Which program offers more flexibility for working professionals?"

4."Compare the technical skills and programming languages covered between both programs."

5."How do the start dates and program durations differ?"
