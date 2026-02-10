# Dockerfile – PyTorch Nightly for sm_120 (No cuDNN Upgrade Needed)
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends git curl && \
    rm -rf /var/lib/apt/lists/*

# Non-root user
ARG USERNAME=appuser
ARG USER_UID=1000
RUN groupadd --gid $USER_UID $USERNAME && \
    useradd --uid $USER_UID --gid $USER_UID -m $USERNAME

# === ROOT: Uninstall stable PyTorch ===
USER root
RUN pip uninstall -y torch torchvision torchaudio || true

# === Back to non-root ===
USER $USERNAME
ENV PATH="/home/${USERNAME}/.local/bin:${PATH}"
ENV LD_LIBRARY_PATH="/opt/conda/lib:${LD_LIBRARY_PATH}"
WORKDIR /workspace

# Install PyTorch nightly (cu121 matches base image's CUDA 12.1 + cuDNN 8)
RUN pip install --no-cache-dir --user --pre \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/cu121

# Install FAISS-GPU
RUN pip install --no-cache-dir --user faiss-gpu-cu12==1.12.0

# Other deps
COPY --chown=$USERNAME:$USERNAME requirements.txt .
RUN pip install --no-cache-dir --user --upgrade pip && \
    pip install --no-cache-dir --user -r requirements.txt

# Verify PyTorch nightly + sm_120 (no cuDNN error)
RUN python - <<'PY'
import torch
print("PyTorch nightly:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Supported archs:", torch.cuda.get_arch_list())
print("GPU count:", torch.cuda.device_count())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(i)
        major, minor = torch.cuda.get_device_capability(i)
        print(f"  GPU {i}: {name} → sm_{major}{minor}")
else:
    print("  No CUDA devices visible")
PY

# Verify FAISS
RUN python -c "import faiss; print('FAISS GPUs:', faiss.get_num_gpus())"

COPY --chown=$USERNAME:$USERNAME . .

EXPOSE 8004
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8004"]
