# rag/index.py

import faiss
import pickle
import numpy as np
from pathlib import Path
from typing import Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredMarkdownLoader,
    TextLoader,
)

from .embeddings import embeddings, useGpu as USE_GPU, gpuIndex as GPU_INDEX_NUM

DOCUMENTS_PATH = Path("./documents")
VECTORSTORE_PATH = Path("./vectorstore")
VECTORSTORE_PATH.mkdir(exist_ok=True)

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
TOP_K = 3

index: Optional[faiss.Index] = None
texts = []
metadatas = []


# ------------------- Index Management -------------------
def load_or_rebuild_index():
    global index, texts, metadatas
    pkl_path = VECTORSTORE_PATH / "index.pkl"
    faiss_path = VECTORSTORE_PATH / "index.faiss"

    if faiss_path.exists() and pkl_path.exists():
        print("Loading existing FAISS index from disk...")
        index = faiss.read_index(str(faiss_path))
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
            texts = data["texts"]
            metadatas = data["metadatas"]
    else:
        print("No index found → rebuilding from ./documents/")
        rebuild_index_from_disk()

def rebuild_index_from_disk():
    global index, texts, metadatas
    docs = []
    for file_path in DOCUMENTS_PATH.rglob("*"):
        if not file_path.is_file():
            continue
        suffix = file_path.suffix.lower()
        if suffix == ".pdf":
            docs.extend(PyPDFLoader(str(file_path)).load())
        elif suffix == ".md":
            docs.extend(UnstructuredMarkdownLoader(str(file_path)).load())
        elif suffix in {".txt", ".py", ".js", ".ts", ".json", ".csv", ".html", ".log"}:
            docs.extend(TextLoader(str(file_path), encoding="utf-8").load())

    if not docs:
        #dim = 384
        dim = 1024
        index = faiss.IndexFlatIP(dim)
        faiss.write_index(index, str(VECTORSTORE_PATH / "index.faiss"))
        with open(VECTORSTORE_PATH / "index.pkl", "wb") as f:
            pickle.dump({"texts": [], "metadatas": []}, f)
        texts, metadatas = [], []
        return

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(docs)
    vectors = np.array(embeddings.embed_documents([c.page_content for c in chunks])).astype("float32")

    dim = vectors.shape[1]
    nlist = max(1, int(np.sqrt(len(chunks))))
    quantizer = faiss.IndexFlatIP(dim)
    cpu_index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)

    # if torch.cuda.is_available():
    if USE_GPU:
        print(f"Index rebuilt: {len(chunks)} chunks on RTX GPU {GPU_INDEX_NUM}")
        res = faiss.StandardGpuResources()
        gpu_index = faiss.index_cpu_to_gpu(res, GPU_INDEX_NUM, cpu_index)
        gpu_index.train(vectors)
        gpu_index.add(vectors)
        index = gpu_index
    else:
        print(f"Index rebuilt: {len(chunks)} chunks on CPU")
        cpu_index.train(vectors)
        cpu_index.add(vectors)
        index = cpu_index

    # Save FAISS index — works for both GPU and CPU indexes
    if getattr(index, "is_gpu", False):  # Modern faiss-gpu-cu12
        faiss.write_index(faiss.index_gpu_to_cpu(index), str(VECTORSTORE_PATH / "index.faiss"))
    elif "GpuIndex" in str(type(index)):  # Fallback for older versions
        faiss_index_cpu = faiss.index_gpu_to_cpu(index)
        faiss.write_index(faiss_index_cpu, str(VECTORSTORE_PATH / "index.faiss"))
    else:
        faiss.write_index(index, str(VECTORSTORE_PATH / "index.faiss"))

    texts = [c.page_content for c in chunks]
    metadatas = [c.metadata for c in chunks]
    with open(VECTORSTORE_PATH / "index.pkl", "wb") as f:
        pickle.dump({"texts": texts, "metadatas": metadatas}, f)

def rebuild_index_from_memory(new_texts, new_metadatas):
    global index, texts, metadatas
    texts = new_texts
    metadatas = new_metadatas

    if not texts:
        #dim = 384
        dim = 1024
        index = faiss.IndexFlatIP(dim)
        faiss.write_index(index, str(VECTORSTORE_PATH / "index.faiss"))
        with open(VECTORSTORE_PATH / "index.pkl", "wb") as f:
            pickle.dump({"texts": [], "metadatas": []}, f)
        return

    vectors = np.array(embeddings.embed_documents(texts)).astype("float32")
    dim = vectors.shape[1]
    nlist = max(1, int(np.sqrt(len(texts))))
    quantizer = faiss.IndexFlatIP(dim)
    cpu_index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)

    # if torch.cuda.is_available():
    if USE_GPU:
        print("Building FAISS index on GPU")
        res = faiss.StandardGpuResources()
        gpu_index = faiss.index_cpu_to_gpu(res, GPU_INDEX_NUM, cpu_index)
        gpu_index.train(vectors)
        gpu_index.add(vectors)
        index = gpu_index
    else:
        print("Building FAISS index on CPU (safe fallback)")
        cpu_index.train(vectors)
        cpu_index.add(vectors)
        index = cpu_index

    # Save FAISS index — works for both GPU and CPU indexes
    if getattr(index, "is_gpu", False):  # Modern faiss-gpu-cu12
        faiss.write_index(faiss.index_gpu_to_cpu(index), str(VECTORSTORE_PATH / "index.faiss"))
    elif "GpuIndex" in str(type(index)):  # Fallback for older versions
        faiss_index_cpu = faiss.index_gpu_to_cpu(index)
        faiss.write_index(faiss_index_cpu, str(VECTORSTORE_PATH / "index.faiss"))
    else:
        faiss.write_index(index, str(VECTORSTORE_PATH / "index.faiss"))

    with open(VECTORSTORE_PATH / "index.pkl", "wb") as f:
        pickle.dump({"texts": texts, "metadatas": metadatas}, f)

    print(f"Memory-based index rebuilt & saved: {len(texts)} chunks")

load_or_rebuild_index()
