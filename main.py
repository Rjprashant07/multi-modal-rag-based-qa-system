"""
main.py
Multi-Modal RAG over PDF documents using:
- HuggingFace CLIP for unified multi-modal embeddings (text + images)
- HuggingFace FLAN-T5 for generation
- ChromaDB (new API, PersistentClient) as vector store
- FastAPI as API layer
"""

import io
import os
import re
import uuid
from typing import List, Dict, Optional

import fitz  # PyMuPDF
from PIL import Image
import pytesseract

import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer, AutoModelForSeq2SeqLM

import chromadb
from chromadb import PersistentClient

# Configuration

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
GEN_MODEL_NAME = "google/flan-t5-base"  # can change to -small if GPU RAM is low

CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "multi_modal_docs"
IMAGE_DIR = "./pdf_images"
os.makedirs(IMAGE_DIR, exist_ok=True)



# Model loader (lazy singletons)

class ModelRegistry:
    _clip_model = None
    _clip_processor = None
    _tokenizer = None
    _gen_model = None

    @classmethod
    def get_clip(cls):
        if cls._clip_model is None or cls._clip_processor is None:
            print("Loading CLIP model:", CLIP_MODEL_NAME)
            cls._clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(DEVICE)
            cls._clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
        return cls._clip_model, cls._clip_processor

    @classmethod
    def get_generator(cls):
        if cls._tokenizer is None or cls._gen_model is None:
            print("Loading generator model:", GEN_MODEL_NAME)
            cls._tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL_NAME)
            cls._gen_model = AutoModelForSeq2SeqLM.from_pretrained(GEN_MODEL_NAME).to(DEVICE)
        return cls._tokenizer, cls._gen_model



# Chroma client & collection (NEW API)


client: PersistentClient = chromadb.PersistentClient(path=CHROMA_DIR)
collection = client.get_or_create_collection(name=COLLECTION_NAME)



# Embedding helpers (CLIP)


def encode_text_clip(texts: List[str]) -> np.ndarray:
    """Encode a list of texts into CLIP text embeddings."""
    clip_model, clip_processor = ModelRegistry.get_clip()
    inputs = clip_processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
    with torch.no_grad():
        text_embs = clip_model.get_text_features(**inputs)
    text_embs = text_embs / text_embs.norm(dim=-1, keepdim=True)
    return text_embs.cpu().numpy()


def encode_images_clip(image_paths: List[str]) -> np.ndarray:
    """Encode a list of image file paths into CLIP image embeddings."""
    clip_model, clip_processor = ModelRegistry.get_clip()
    images = [Image.open(p).convert("RGB") for p in image_paths]
    inputs = clip_processor(images=images, return_tensors="pt", padding=True).to(DEVICE)
    with torch.no_grad():
        img_embs = clip_model.get_image_features(**inputs)
    img_embs = img_embs / img_embs.norm(dim=-1, keepdim=True)
    return img_embs.cpu().numpy()



# PDF ingestion: text + images + OCR + simple captions


def extract_page_text(page) -> str:
    """Simple full-page text extraction."""
    return page.get_text("text")


def find_chart_captions(page_text: str) -> List[str]:
    """Find lines that look like figure/chart captions."""
    captions = []
    for line in page_text.splitlines():
        if re.match(r"^\s*(Figure|Chart|Exhibit)\b", line.strip(), flags=re.IGNORECASE):
            captions.append(line.strip())
    return captions


def ingest_pdf_bytes(doc_id: str, file_bytes: bytes) -> List[Dict]:
    """
    Ingest a single PDF from bytes.
    Returns a list of 'items', each with modality and metadata BEFORE chunking/embedding.
    """
    items: List[Dict] = []
    doc = fitz.open(stream=file_bytes, filetype="pdf")

    for page_index in range(len(doc)):
        page = doc[page_index]
        page_number = page_index + 1
        page_text = extract_page_text(page)
        page_text = page_text.strip()

        section_title = f"Page {page_number}"  # simple section label

        # ---- TEXT ITEM ----
        if page_text:
            items.append({
                "id": f"{doc_id}_text_p{page_number}",
                "doc_id": doc_id,
                "modality": "text",
                "content": page_text,
                "page_number": page_number,
                "section": section_title,
                "extra": {},
            })

        # ---- IMAGE ITEMS ----
        image_list = page.get_images(full=True)
        captions = find_chart_captions(page_text)
        caption_str = " | ".join(captions) if captions else ""

        for img_idx, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

            img_filename = f"{doc_id}_page{page_number}_img{img_idx}.png"
            img_path = os.path.join(IMAGE_DIR, img_filename)
            pil_img.save(img_path)

            ocr_text = pytesseract.image_to_string(pil_img)

            items.append({
                "id": f"{doc_id}_img_p{page_number}_{img_idx}",
                "doc_id": doc_id,
                "modality": "image",
                "content": ocr_text.strip(),  # used in LLM context
                "page_number": page_number,
                "section": section_title,
                "extra": {
                    "image_path": img_path,
                    "caption": caption_str,
                },
            })

    doc.close()
    return items



# Smart Hybrid Chunking (Semantic + Structural)


def structural_split(text: str) -> List[str]:
    """
    Detect structural boundaries (section headers, numbered headings, all caps, etc.).
    Returns a list of larger blocks split by structure.
    """
    lines = text.split("\n")
    chunks = []
    buffer = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            buffer.append(line)
            continue

        is_heading = (
            re.match(r"^\d+(\.\d+)*\s", stripped) or         # 1.  / 1.1 / 2.3.4
            re.match(r"^[A-Z][A-Z\s\-]{6,}$", stripped) or   # ALL CAPS headings
            re.match(r"^[A-Za-z].+:\s*$", stripped) or       # Title:
            re.match(r"^(Conclusion|Summary|Overview)\b", stripped, re.I)
        )

        # New section detected → flush buffer
        if is_heading and buffer:
            chunks.append("\n".join(buffer))
            buffer = [line]
        else:
            buffer.append(line)

    if buffer:
        chunks.append("\n".join(buffer))

    return chunks


def semantic_chunk_block(text: str, max_tokens: int = 1200, threshold: float = 0.65) -> List[str]:
    """
    Split a text block into semantic chunks using sentence embeddings.
    """
    from nltk.tokenize import sent_tokenize

    sentences = sent_tokenize(text)
    if len(sentences) <= 1:
        return [text]

    # Encode each sentence with CLIP embeddings
    embeddings = encode_text_clip(sentences)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    chunks = []
    current = [sentences[0]]
    current_len = len(sentences[0])

    for i in range(1, len(sentences)):
        sim = float(np.dot(embeddings[i], embeddings[i - 1]))

        # Low similarity OR chunk too long → start new chunk
        if sim < threshold or current_len > max_tokens:
            chunks.append(" ".join(current))
            current = [sentences[i]]
            current_len = len(sentences[i])
        else:
            current.append(sentences[i])
            current_len += len(sentences[i])

    # final chunk
    chunks.append(" ".join(current))
    return chunks


def chunk_text(text: str) -> List[str]:
    """
    Hybrid chunking:
    1. Split using document structure (headings, sections).
    2. Within each block, perform semantic chunking.
    """
    structural_blocks = structural_split(text)
    final_chunks: List[str] = []

    for block in structural_blocks:
        block = block.strip()
        if not block:
            continue
        sem_chunks = semantic_chunk_block(block)
        final_chunks.extend(sem_chunks)

    return final_chunks


def items_to_chunks(items: List[Dict]) -> List[Dict]:
    """
    Convert per-page items into smaller chunks for embedding.
    Text → hybrid (structural + semantic).
    Image → OCR + caption stays together.
    """
    chunks: List[Dict] = []

    for it in items:
        if it["modality"] == "text":
            text_chunks = chunk_text(it["content"])
            for idx, ch in enumerate(text_chunks):
                chunks.append({
                    "id": f"{it['id']}_chunk{idx}",
                    "doc_id": it["doc_id"],
                    "modality": "text",
                    "content": ch,
                    "page_number": it["page_number"],
                    "section": it.get("section") or "",
                    "extra": {},  # not used for text in metadata; keep flat
                })
        elif it["modality"] == "image":
            combined = it["content"]
            caption = it["extra"].get("caption")
            if caption:
                combined = f"{caption}\n\n{combined}"

            chunks.append({
                "id": it["id"],
                "doc_id": it["doc_id"],
                "modality": "image",
                "content": combined,
                "page_number": it["page_number"],
                "section": it.get("section") or "",
                "extra": it.get("extra", {}),
            })

    return chunks



# Indexing (Chroma + CLIP)


def index_chunks(chunks: List[Dict]):
    """
    Index all chunks in unified multi-modal space.
    Text-like: CLIP text encoder.
    Images: CLIP image encoder, document string is OCR+caption.
    """
    if not chunks:
        return

    text_like = [c for c in chunks if c["modality"] != "image"]
    image_like = [c for c in chunks if c["modality"] == "image"]

    # Text-like chunks
    if text_like:
        texts = [c["content"] for c in text_like]
        text_embs = encode_text_clip(texts)
        collection.add(
            ids=[c["id"] for c in text_like],
            embeddings=text_embs.tolist(),
            documents=texts,
            metadatas=[
                {
                    "doc_id": str(c["doc_id"]),
                    "modality": str(c["modality"]),
                    "page_number": int(c["page_number"]),
                    "section": str(c.get("section") or ""),
                    # text chunks don't have image data
                    "image_path": "",
                    "caption": "",
                }
                for c in text_like
            ],
        )

    # Image chunks
    if image_like:
        image_paths = [c.get("extra", {}).get("image_path", "") for c in image_like]
        img_embs = encode_images_clip(image_paths)
        docs_for_llm = [c["content"] for c in image_like]  # OCR+caption
        collection.add(
            ids=[c["id"] for c in image_like],
            embeddings=img_embs.tolist(),
            documents=docs_for_llm,
            metadatas=[
                {
                    "doc_id": str(c["doc_id"]),
                    "modality": str(c["modality"]),
                    "page_number": int(c["page_number"]),
                    "section": str(c.get("section") or ""),
                    "image_path": c.get("extra", {}).get("image_path", ""),
                    "caption": c.get("extra", {}).get("caption", ""),
                }
                for c in image_like
            ],
        )

    print(f"Indexed {len(chunks)} chunks (text-like={len(text_like)}, image={len(image_like)})")

# -----------------------------
# Retrieval & RAG
# -----------------------------

def retrieve_chunks(query: str, doc_id: str, top_k: int = 8) -> List[Dict]:
    """Retrieve top-k multi-modal chunks for a given query and doc_id."""
    q_emb = encode_text_clip([query])
    results = collection.query(
        query_embeddings=q_emb.tolist(),
        n_results=top_k,
        where={"doc_id": doc_id},
    )

    if not results["ids"]:
        return []

    retrieved = []
    for i in range(len(results["ids"][0])):
        retrieved.append({
            "id": results["ids"][0][i],
            "content": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i] if "distances" in results else None,
        })
    return retrieved


def format_context_for_prompt(chunks: List[Dict]) -> str:
    """Produce LLM context string with explicit source tags."""
    blocks = []
    for idx, c in enumerate(chunks, start=1):
        m = c["metadata"]
        page = m.get("page_number", "?")
        section = m.get("section", "Unknown section")
        modality = m.get("modality", "text")
        tag = f"[SOURCE {idx} – page {page}, section: {section}, modality: {modality}]"
        content = c["content"].strip()
        if len(content) > 1200:
            content = content[:1200] + "... [truncated]"
        blocks.append(f"{tag}\n{content}\n")
    return "\n\n".join(blocks)


def build_rag_prompt(question: str, chunks: List[Dict]) -> str:
    context = format_context_for_prompt(chunks)
    prompt = f"""
You are a QA assistant for IMF-style financial and policy documents.

You are given context extracted from:
- Text paragraphs
- Tables (as text)
- Images and charts (represented via OCR text and captions)

Context:
{context}

Instructions:
- Answer the user's question ONLY using the context above.
- If the answer is not in the context, say you don't know.
- Always include citations using the exact tags like [SOURCE i – page X, section: Y, modality: Z].

Question: {question}

Answer:
"""
    # Trim if too long
    if len(prompt) > 6000:
        prompt = prompt[-6000:]
    return prompt


def call_llm(prompt: str, max_new_tokens: int = 256) -> str:
    tokenizer, gen_model = ModelRegistry.get_generator()
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    ).to(DEVICE)

    with torch.no_grad():
        outputs = gen_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=4,
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def rag_answer(question: str, doc_id: str, top_k: int = 8) -> Dict:
    chunks = retrieve_chunks(question, doc_id, top_k=top_k)
    if not chunks:
        return {
            "answer": "I couldn't find any relevant context for this document.",
            "sources": [],
        }
    prompt = build_rag_prompt(question, chunks)
    answer = call_llm(prompt)
    return {
        "answer": answer,
        "sources": chunks,
    }


# -----------------------------
# FastAPI models & app
# -----------------------------

class QueryRequest(BaseModel):
    doc_id: str
    question: str
    top_k: Optional[int] = 8


class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict]


class IngestResponse(BaseModel):
    doc_id: str
    num_items: int
    num_chunks: int


app = FastAPI(title="Multi-Modal RAG over Documents")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"status": "ok", "message": "Multi-Modal RAG API is running."}


@app.post("/ingest", response_model=IngestResponse)
async def ingest_endpoint(
    file: UploadFile = File(...),
    doc_id: Optional[str] = Form(None),
):
    """Ingest a PDF into the RAG system."""
    if doc_id is None or not doc_id.strip():
        base = os.path.splitext(file.filename)[0] if file.filename else "doc"
        doc_id = f"{base}_{uuid.uuid4().hex[:8]}"

    file_bytes = await file.read()

    items = ingest_pdf_bytes(doc_id, file_bytes)
    chunks = items_to_chunks(items)
    index_chunks(chunks)

    return IngestResponse(
        doc_id=doc_id,
        num_items=len(items),
        num_chunks=len(chunks),
    )


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(req: QueryRequest):
    """Ask a question about a previously ingested document."""
    result = rag_answer(req.question, req.doc_id, top_k=req.top_k or 8)
    return QueryResponse(
        answer=result["answer"],
        sources=result["sources"],
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)


