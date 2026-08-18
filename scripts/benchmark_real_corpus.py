import argparse
import time
from pathlib import Path

import faiss
import numpy as np
import torch
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


MODEL = "BAAI/bge-large-en-v1.5"
DOCUMENTS_PATH = Path("./documents")
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
MIN_TRAINING_POINTS_PER_CENTROID = 39


def choose_nlist(num_vectors: int) -> int:
    if num_vectors <= 0:
        return 1

    raw_nlist = max(1, int(np.sqrt(num_vectors)))
    capped_nlist = max(1, num_vectors // MIN_TRAINING_POINTS_PER_CENTROID)
    final_nlist = min(raw_nlist, capped_nlist)

    if final_nlist < raw_nlist:
        print(
            f"Auto-capping nlist from {raw_nlist} to {final_nlist} "
            f"for {num_vectors} training vectors"
        )

    return final_nlist


def build_ivf_index(vectors: np.ndarray, nlist: int, use_faiss_gpu: bool):
    dim = vectors.shape[1]
    quantizer = faiss.IndexFlatIP(dim)
    cpu_index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)

    if use_faiss_gpu:
        try:
            faiss_gpu_count = faiss.get_num_gpus() if hasattr(faiss, "get_num_gpus") else 0
            if faiss_gpu_count <= 0:
                raise RuntimeError("No FAISS GPUs are visible")

            res = faiss.StandardGpuResources()
            gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
            gpu_index.train(vectors)
            gpu_index.add(vectors)
            return gpu_index, "gpu"
        except Exception as exc:
            print(f"FAISS GPU path failed in benchmark: {exc}; falling back to CPU")

    cpu_index.train(vectors)
    cpu_index.add(vectors)
    return cpu_index, "cpu"


def load_documents() -> list:
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

    return docs


def run_benchmark(profile: str, texts: list[str]) -> dict:
    if profile == "cpu":
        device = "cpu"
        use_faiss_gpu = False
        require_faiss_gpu = False
    elif profile == "gpu":
        device = "cuda:0"
        use_faiss_gpu = True
        require_faiss_gpu = True
    else:
        raise ValueError(f"Unsupported profile: {profile}")

    embeddings = HuggingFaceEmbeddings(
        model_name=MODEL,
        encode_kwargs={"normalize_embeddings": True},
        model_kwargs={"device": device},
    )

    embeddings.embed_documents(texts[:8])

    start_embed = time.perf_counter()
    vectors = np.array(embeddings.embed_documents(texts), dtype="float32")
    end_embed = time.perf_counter()

    dim = vectors.shape[1]
    nlist = choose_nlist(len(texts))

    start_faiss = time.perf_counter()
    _index, faiss_mode = build_ivf_index(vectors, nlist, use_faiss_gpu=use_faiss_gpu)
    end_faiss = time.perf_counter()

    if require_faiss_gpu and faiss_mode != "gpu":
        raise RuntimeError("GPU profile requires FAISS GPU indexing, but benchmark fell back to CPU")

    embed_seconds = end_embed - start_embed
    faiss_seconds = end_faiss - start_faiss

    return {
        "profile": profile,
        "device": device,
        "chunks": len(texts),
        "dim": dim,
        "embed_s": embed_seconds,
        "faiss_s": faiss_seconds,
        "faiss_mode": faiss_mode,
        "total_s": embed_seconds + faiss_seconds,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark real corpus indexing on CPU and/or GPU")
    parser.add_argument(
        "--profile",
        choices=["cpu", "gpu", "both"],
        default="both",
        help="Benchmark profile to run",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("torch", torch.__version__)
    print("cuda_available", torch.cuda.is_available())
    print("cuda_count", torch.cuda.device_count())
    print("faiss", faiss.__version__)

    docs = load_documents()
    print("documents_loaded", len(docs))
    if not docs:
        print("No documents found under ./documents")
        return

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(docs)
    texts = [c.page_content for c in chunks if c.page_content and c.page_content.strip()]
    print("chunks", len(texts))

    if not texts:
        print("No chunk text produced from corpus")
        return

    cpu_result = None
    gpu_result = None

    if args.profile in {"cpu", "both"}:
        cpu_result = run_benchmark("cpu", texts)
        print("CPU", cpu_result)

    if args.profile in {"gpu", "both"}:
        if torch.cuda.is_available():
            gpu_result = run_benchmark("gpu", texts)
            print("GPU", gpu_result)
        else:
            raise RuntimeError("GPU profile requested, but CUDA is unavailable")

    if cpu_result and gpu_result:
        print("speedup_embed_x", round(cpu_result["embed_s"] / gpu_result["embed_s"], 2))
        print("speedup_total_x", round(cpu_result["total_s"] / gpu_result["total_s"], 2))


if __name__ == "__main__":
    main()
