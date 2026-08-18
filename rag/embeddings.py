# rag/embeddings.py

import os
import gc
import subprocess
import sys
import torch
from langchain_huggingface import HuggingFaceEmbeddings

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_MODEL_LARGE = "BAAI/bge-large-en-v1.5"

RAG_USE_GPU = os.getenv("RAG_USE_GPU", "false").strip().lower() in {"1", "true", "yes", "on"}
RAG_GPU_INDEX_RAW = os.getenv("RAG_GPU_INDEX", "").strip()

useGpu = False
gpuIndex = 0


def _is_gpu_usable(candidate_index: int) -> bool:
    test_code = (
        "import numpy\n"
        "import torch\n"
        f"torch.zeros(1, device='cuda:{candidate_index}')\n"
        "print('ok')\n"
    )

    try:
        result = subprocess.run(
            [sys.executable, "-c", test_code],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=20,
        )
    except Exception as e:
        print(f"Skipping GPU {candidate_index}: subprocess CUDA test error: {e}")
        return False

    if result.returncode == 0:
        return True

    stderr = (result.stderr or "").strip().splitlines()
    last_err = stderr[-1] if stderr else "unknown CUDA test failure"
    print(f"Skipping GPU {candidate_index}: subprocess CUDA test failed: {last_err}")
    return False

# -----------------------------
# GPU selection + diagnostics
# -----------------------------
if RAG_USE_GPU:
    if torch.cuda.is_available():
        try:
            device_count = torch.cuda.device_count()
            print("PyTorch version:", torch.__version__)
            print("CUDA available:", torch.cuda.is_available())
            print("GPU count:", device_count)

            if device_count > 0:
                candidate_indices = []
                if RAG_GPU_INDEX_RAW:
                    try:
                        requested_index = int(RAG_GPU_INDEX_RAW)
                        if 0 <= requested_index < device_count:
                            candidate_indices.append(requested_index)
                        else:
                            print(
                                f"Requested GPU index {requested_index} is out of range; "
                                "trying all visible GPUs"
                            )
                    except ValueError:
                        print(f"Invalid RAG_GPU_INDEX '{RAG_GPU_INDEX_RAW}' — trying all visible GPUs")

                candidate_indices.extend(i for i in range(device_count) if i not in candidate_indices)

                for candidate_index in candidate_indices:
                    if _is_gpu_usable(candidate_index):
                        gpuIndex = candidate_index
                        useGpu = True
                        selected_name = torch.cuda.get_device_name(gpuIndex)
                        print(f"Using GPU {gpuIndex}: {selected_name}")
                        break

                if not useGpu:
                    print("No usable CUDA GPU found — falling back to CPU")
            else:
                print("No CUDA devices found — falling back to CPU")

        except Exception as e:
            print("CUDA available but GPU selection failed:", e)
    else:
        print("No CUDA — CPU mode")

    # -----------------------------
    # Embedding model initialization
    # -----------------------------
    device = f"cuda:{gpuIndex}" if useGpu else "cpu"
else:
    print("GPU usage disabled — CPU mode")
    device = "cpu"

EMBEDDING_DEVICE = device

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_LARGE,
    encode_kwargs={"normalize_embeddings": True}, # normalize to unit length, only for EMBEDDING_MODEL_LARGE
    model_kwargs={"device": EMBEDDING_DEVICE}
)


def cleanup_after_embedding() -> None:
    gc.collect()

    if not EMBEDDING_DEVICE.startswith("cuda"):
        return

    if not torch.cuda.is_available():
        return

    torch.cuda.empty_cache()
    if hasattr(torch.cuda, "ipc_collect"):
        torch.cuda.ipc_collect()
