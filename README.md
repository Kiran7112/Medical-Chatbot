# 📌 End-to-end Medical Chatbot using Llama2

![Python](https://img.shields.io/badge/Python-3.8-blue?style=for-the-badge&logo=python)
![Flask](https://img.shields.io/badge/Flask-Framework-black?style=for-the-badge&logo=flask)
![Pinecone](https://img.shields.io/badge/Pinecone-VectorDB-blue?style=for-the-badge)
![LangChain](https://img.shields.io/badge/LangChain-LLM-orange?style=for-the-badge)

## 🚀 How to Run?

### 🔹 Steps:

📌 **Clone the repository**
```bash
Project repo: https://github.com/
```

### 🔹 Step 01 - Create a Conda Environment
```bash
conda create -n mchatbot python=3.8 -y
conda activate mchatbot
```

### 🔹 Step 02 - Install the Requirements
```bash
pip install -r requirements.txt
```

### 🔹 Step 03 - Setup Pinecone Credentials
Create a `.env` file in the root directory and add the following:
```ini
PINECONE_API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
PINECONE_API_ENV = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

### 🔹 Step 04 - Download the Quantized Model
📥 Download **llama-2-7b-chat.ggmlv3.q4_0.bin** from the link below and place it in the `model` directory:
[Download Llama2 Model](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/tree/main)

### 🔹 Step 05 - Store the Index
```bash
python store_index.py
```

### 🔹 Step 06 - Run the Application
```bash
python app.py
```

📌 Open up **localhost** to interact with the chatbot.

## 🖼️ Screenshots
 ![Screenshot 2025-02-26 195249](https://github.com/user-attachments/assets/e9bc5bd6-f65d-4a8b-ad96-735a33f5e916)

### 🔹 Home Page
![Home Page](screenshots/home.png)

### 🔹 Chat Interface
![Chat Interface](screenshots/chat.png)

## 🛠️ Tech Stack Used

- ![Python](https://img.shields.io/badge/Python-3.8-blue?style=flat&logo=python) **Python**
- ![LangChain](https://img.shields.io/badge/LangChain-LLM-orange?style=flat) **LangChain**
- ![Flask](https://img.shields.io/badge/Flask-Framework-black?style=flat&logo=flask) **Flask**
- ![Meta](https://img.shields.io/badge/Meta-Llama2-lightgray?style=flat) **Meta Llama2**
- ![Pinecone](https://img.shields.io/badge/Pinecone-VectorDB-blue?style=flat) **Pinecone**

---
💡 *Contributions are welcome! Feel free to submit a PR.*

