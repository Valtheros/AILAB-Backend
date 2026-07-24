FROM python:3.11-slim

WORKDIR /app

# Install Python dependencies before copying source so Docker can cache the layer.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# CPU-only PyTorch for single-image model testing (/api/runs/{slug}/predict).
# Versions match the training worker so saved state_dicts load unchanged.
# The CPU wheels are installed from PyTorch's own index because the default
# PyPI wheels bundle CUDA and would add several gigabytes to this image; the
# GPU is deliberately left to the training worker so inference cannot compete
# for VRAM with an active training job.
#
# numpy is pinned below 2.0 for the same reason as the training worker: torch
# 2.2 is built against the numpy 1.x ABI, and torchvision's ToTensor() fails at
# runtime with "Numpy is not available" when numpy 2.x is installed.
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple \
    torch==2.2.0 torchvision==0.17.0 "numpy<2"

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
