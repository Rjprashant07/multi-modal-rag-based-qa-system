# 📄 Multi-Modal RAG Based QA System
**Retrieval-Augmented Generation over PDFs with Text, Images, OCR, and Semantic Chunking**

This project implements a **Multi-Modal Retrieval-Augmented Generation (RAG) system** capable of answering questions from complex documents such as IMF Article IV reports, financial analyses, research papers, and policy documents. The system combines **text**, **images**, **OCR**, **captions**, and **semantic chunking** to extract meaningful insights and generate grounded, citation-backed answers.

This project is built using **FastAPI**, **ChromaDB**, **HuggingFace CLIP**, **FLAN-T5**, and **PyMuPDF**, and is implemented **without LangChain** for full transparency and control.

---

## 🚀 Features

### 🔹 Multi-Modal Document Ingestion
- Extracts text from PDF pages  
- Extracts images and performs OCR using Tesseract  
- Identifies figure/chart captions  
- Embeds images using CLIP  
- Stores page-level and section-level metadata  

### 🔹 Smart Hybrid Chunking (Structural + Semantic)
| Chunking Type | Description |
|---------------|-------------|
| **Structural Chunking** | Splits based on headings, numbered sections, ALL CAPS content, captions |
| **Semantic Chunking** | Uses CLIP embeddings to split text based on semantic similarity |
| **Multi-Modal Chunking** | Combines OCR results, captions, and image embeddings |

### 🔹 Unified Multi-Modal Embeddings
Uses CLIP to embed:
- Text (CLIP text encoder)  
- Images (CLIP image encoder)  

Both exist in the **same vector space**, enabling semantic cross-modal retrieval.

### 🔹 Vector-Based Retrieval (ChromaDB)
- Persistent vector database  
- Cosine similarity search  
- Top-k relevant chunk retrieval  
- Metadata filtering by `doc_id`  

### 🔹 LLM Answer Generation (FLAN-T5)
- Creates a custom prompt including retrieved chunks  
- Generates grounded answers  
- Adds citations:

### 🔹 FastAPI Backend + Web Interface
Endpoints:
- `POST /ingest` → Upload and process a PDF  
- `POST /query` → Query the document  
- `GET /qa` → Interactive browser-based QA UI  

---

## 📦 Installation & Setup

### **1. Install Dependencies**
```bash
pip install -r requirements.txt

### **2.Run the FastAPI Server**

uvicorn main:app --host 0.0.0.0 --port 8000


3. Access the Application

API Root → http://localhost:8000/

Web QA Interface → http://localhost:8000/qa

---

  
