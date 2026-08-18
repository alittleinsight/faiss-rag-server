import time
import numpy as np
import torch
import faiss

from langchain_huggingface import HuggingFaceEmbeddings


MODEL = "BAAI/bge-large-en-v1.5"
NUM_TEXTS = 600


def build_texts(n: int):
    return [(f"This is synthetic benchmark chunk {i}. ") * 40 for i in range(n)]


def run_benchmark(device: str, texts: list[str]) -> dict:
    embeddings = HuggingFaceEmbeddings(
        model_name=MODEL,
        encode_kwargs={"normalize_embeddings": True},
        model_kwargs={"device": device},
    )

    embeddings.embed_documents(texts[:16])

    start_embed = time.perf_counter()
    vectors = np.array(embeddings.embed_documents(texts), dtype="float32")
    end_embed = time.perf_counter()

    dim = vectors.shape[1]
    nlist = max(1, int(len(texts) ** 0.5))
    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)

    start_faiss = time.perf_counter()
    index.train(vectors)
    index.add(vectors)
    end_faiss = time.perf_counter()

    embed_seconds = end_embed - start_embed
    faiss_seconds = end_faiss - start_faiss
    total_seconds = embed_seconds + faiss_seconds

    return {
        "device": device,
        "vectors": len(vectors),
        "dim": dim,
        "embed_s": embed_seconds,
        "faiss_cpu_s": faiss_seconds,
        "total_s": total_seconds,
    }


def main():
    texts = build_texts(NUM_TEXTS)

    print("torch", torch.__version__)
    print("cuda_available", torch.cuda.is_available())
    print("cuda_count", torch.cuda.device_count())
    print("faiss", faiss.__version__)

    cpu = run_benchmark("cpu", texts)
    print("CPU", cpu)

    if torch.cuda.is_available():
        gpu = run_benchmark("cuda:0", texts)
        print("GPU", gpu)
        print("speedup_embed_x", round(cpu["embed_s"] / gpu["embed_s"], 2))
        print("speedup_total_x", round(cpu["total_s"] / gpu["total_s"], 2))
    else:
        print("GPU unavailable; skipped GPU benchmark")


if __name__ == "__main__":
    main()
