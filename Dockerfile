# Dockerfile – Stable PyTorch runtime with FAISS GPU
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl && \
    rm -rf /var/lib/apt/lists/*

# Create non-root user
ARG USERNAME=appuser
ARG USER_UID=1000
RUN groupadd --gid $USER_UID $USERNAME && \
    useradd --uid $USER_UID --gid $USER_UID -m $USERNAME

# Switch back to non-root user
USER $USERNAME
ENV PATH="/home/${USERNAME}/.local/bin:${PATH}"
WORKDIR /workspace

# Install FAISS CPU (stable); embeddings still use GPU via PyTorch
#RUN pip install --no-cache-dir --user faiss-cpu==1.8.0.post1
RUN pip install --no-cache-dir 'faiss-gpu-cu12[fix-cuda]'

# Install Python dependencies
COPY --chown=$USERNAME:$USERNAME requirements.txt .
RUN pip install --no-cache-dir --user --upgrade pip && \
    pip install --no-cache-dir --user -r requirements.txt

# Verify PyTorch and FAISS installation
RUN python - <<'PY'
import torch
print("PyTorch:", torch.__version__)
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

RUN python - <<'PY'
import faiss
print('FAISS import OK:', faiss.__version__)
PY

# Copy application files
COPY --chown=$USERNAME:$USERNAME . .

# Expose application port
EXPOSE 8004

# Default command to run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8004"]
