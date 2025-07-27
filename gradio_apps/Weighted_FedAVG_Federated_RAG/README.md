---
title: Weighted_FedAVG_Federated_RAG
emoji: 📊
colorFrom: yellow
colorTo: yellow
sdk: gradio
sdk_version: 5.38.2
app_file: app.py
pinned: false
license: apache-2.0
short_description: 'Ask questions about online MSc AI at Uni of Hull and Keele '
---
# Weighted FedAVG_Federated_RAG

**Privacy-Preserving Federated Learning Chatbot for MSc AI Program Comparison (Hull vs Keele)**  

The **FedAvg Federated RAG Chatbot** combines **Federated Averaging (FedAvg)** with **Retrieval-Augmented Generation (RAG)** to enable privacy-preserving collaboration across universities. It compares MSc Artificial Intelligence programs at the University of Hull and Keele University without sharing raw institutional data.

If one university's knowledge base is significantly larger/contains more document chunks than the other, consider weighting it proportionally to its data size. 
This allow the larger dataset to have a greater influence on the overall system's

# Weighted FedAvg Federated RAG Chatbot

A privacy-preserving chatbot comparing MSc AI Online Programs at the **University of Hull** and **Keele University**.  
It uses **Weighted Federated Averaging (FedAvg)** with **Retrieval-Augmented Generation (RAG)** to train collaboratively without sharing raw data, while letting you **adjust each university’s influence** using weight sliders.

## How to Use

1. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

2. Data in `./data/hull.txt` and `./data/keele.txt` (sample data is included).

3. Run the app:
   ```bash
   python Weighted_FedAVG_Federated_RAG.py
   ```

4. Use OpenAI API key, set weights, click **Initialize**, and ask questions


## Sample Questions

- 1.Compare the total program costs between MSc Artificial Intelligence at University of Hull online and Keele University online.

- 2.What are the entry requirement differences between MSc Artificial Intelligence at University of Hull online and Keele University?

- 3.Which program offers more flexibility for working professionals?

- 4.Compare the technical skills and programming languages covered between both programs.

- 5.How do the start dates and program durations differ?

- 

The app shows **node-specific answers, confidence scores, and a synthesized response**, factoring in your chosen weights.
