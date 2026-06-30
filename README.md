# RAG PDF Question Answering System

A Retrieval-Augmented Generation (RAG) application that allows users to upload PDF documents and ask natural language questions using AI. The application retrieves relevant document chunks from a vector database and generates context-aware answers using Groq's Llama 3.3 model.

## 🚀 Live Demo

**Render:** https://rag-pdf-question-answering-u9zf.onrender.com/?embed=true

## 📌 Features

- Upload PDF documents
- Extract and chunk PDF text
- Generate embeddings using Sentence Transformers
- Store embeddings in ChromaDB
- Semantic similarity search
- AI-powered question answering using Groq Llama 3.3
- Interactive Streamlit interface

## 🛠️ Tech Stack

- Python
- Streamlit
- FastAPI
- ChromaDB
- Sentence Transformers
- Groq API
- LlamaIndex
- Inngest
- Requests

## 📂 Project Structure

```
RAG-PDF-Question-Answering/
│── streamlit_app.py
│── main.py
│── data_loader.py
│── vector_db.py
│── custom_types.py
│── requirements.txt
│── uploads/
│── chroma_db/
```

## ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/Visuvanathan-K/RAG-PDF-Question-Answering.git
```

Move into the project

```bash
cd RAG-PDF-Question-Answering
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run streamlit_app.py
```

## 💡 How It Works

1. Upload a PDF document.
2. The PDF is split into smaller text chunks.
3. Sentence Transformers generate embeddings for each chunk.
4. Embeddings are stored in ChromaDB.
5. User enters a question.
6. Relevant chunks are retrieved using semantic search.
7. Groq Llama 3.3 generates an answer using the retrieved context.

## 📸 Live Application

https://rag-pdf-question-answering-u9zf.onrender.com/?embed=true

## 👨‍💻 Author

**Visuvanathan K**

GitHub: https://github.com/Visuvanathan-K
