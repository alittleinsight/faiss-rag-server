# rag/embeddings.py

import torch
from langchain_huggingface import HuggingFaceEmbeddings

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_MODEL_LARGE = "BAAI/bge-large-en-v1.5"
SUPPORTED_COMPUTE_CAPABILITY = (7, 5)  # example: sm_75
GPU_AVAILABLE = True
useGpu = False
gpuIndex = 0

# -----------------------------
# GPU selection + diagnostics
# -----------------------------
if GPU_AVAILABLE:
    if torch.cuda.is_available():
        try:
            device_count = torch.cuda.device_count()
            print("PyTorch version:", torch.__version__)
            print("CUDA available:", torch.cuda.is_available())
            print("Supported archs:", torch.cuda.get_arch_list())
            print("GPU count:", device_count)

            for i in range(device_count):
                major, minor = torch.cuda.get_device_capability(i)
                name = torch.cuda.get_device_name(i)
                print(f"  GPU {i}: {name} → sm_{major}{minor}")

                if (major, minor) == SUPPORTED_COMPUTE_CAPABILITY:
                    print(f"Using GPU {i}: {name} (sm_{major}{minor})")
                    gpuIndex = i
                    useGpu = True
                    break
            else:
                print("No supported GPU found — falling back to CPU")

        except Exception as e:
            print("CUDA available but capability check failed:", e)
    else:
        print("No CUDA — CPU mode")

    # -----------------------------
    # Embedding model initialization
    # -----------------------------
    device = f"cuda:{gpuIndex}" if useGpu else "cpu"
else:
    print("GPU usage disabled — CPU mode")
    device = "cpu"

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_LARGE,
    encode_kwargs={"normalize_embeddings": True}, # normalize to unit length, only for EMBEDDING_MODEL_LARGE
    model_kwargs={"device": device}
)
