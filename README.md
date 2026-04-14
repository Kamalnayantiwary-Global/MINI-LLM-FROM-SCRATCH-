# 🧠 Mini LLM from Scratch
made by~
KAMAL NAYAN TIWARY 


![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python: 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg?logo=python&logoColor=white)
![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-ee4c2c.svg?logo=pytorch&logoColor=white)
![Field: NLP / GenAI](https://img.shields.io/badge/Field-NLP%20%2F%20GenAI-brightgreen.svg)
<br>

A lightweight, high-performance **Decoder-only Transformer** architecture (GPT-style) built entirely from scratch using PyTorch. This project demonstrates the fundamental mechanics of Large Language Models, including Multi-Head Attention and Positional Encodings.

---

## 🚀 Overview
This repository contains a character-level language model that learns to generate text by predicting the next token in a sequence. It’s a perfect deep-dive into how models like GPT-4 work under the hood.

### ✨ Key Features
* **Custom Transformer Architecture**: Implemented with Self-Attention, Layer Normalization, and Residual Connections.
* **Scalable Configuration**: Tweak model depth (`n_layer`), width (`n_embd`), and heads (`n_head`) in `config.py`.
* **Efficient Training**: Optimized training loop with loss tracking and periodic evaluation.
* **Inference Script**: Interactive `chat.py` to generate text from a trained model checkpoint.

---

## 🛠️ Tech Stack
* **Language:** Python 3.10+
* **Deep Learning:** PyTorch
* **Tools:** Git, VS Code

---

## 📂 Project Structure
* `model.py` - Core Transformer architecture logic.
* `train.py` - Script for training the model on custom datasets.
* `chat.py` - Script for real-time text generation/inference.
* `config.py` - Hyperparameters and model settings.
* `data.py` - Data loading and character-level encoding/decoding.

---

## 🚦 Getting Started

1. **Clone the Repo**
   ```bash
   git clone [https://github.com/Kamalnayantiwary-Global/MINI-LLM-FROM-SCRATCH-.git](https://github.com/Kamalnayantiwary-Global/MINI-LLM-FROM-SCRATCH-.git) cd MINI-LLM-FROM-SCRATCH-
2. **Install Dependencies**
    pip install torch
3. **Train the model**
    python train.py
4. **Start Generating**
    python chat.py
👨‍💻 Author
​Kamal Nayan Tiwary B.Tech (RCET) | AIML Enthusiast | DSA 
​If you find this project helpful, feel free to ⭐ the repository!
   
  
