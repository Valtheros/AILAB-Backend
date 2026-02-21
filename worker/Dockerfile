# ── Base Image ───────────────────────────────────────────────────────
# ใช้ PyTorch official image ที่มี CUDA มาพร้อมแล้ว — ไม่ต้องติดตั้งเอง
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# ── System Dependencies ──────────────────────────────────────────────
# ติดตั้งครั้งเดียว → cached เป็น image layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# ── Python Dependencies ──────────────────────────────────────────────
# COPY requirements.txt ก่อนแล้ว pip install → Docker จะ cache layer นี้
# ไว้ → rebuild เร็วมากถ้าโค้ดเปลี่ยนแต่ deps ไม่เปลี่ยน
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Application Code ─────────────────────────────────────────────────
COPY . .

# ── Run RQ Worker ────────────────────────────────────────────────────
# Worker จะ subscribe queue ชื่อ "cv_training" และรอรับ job
CMD ["rq", "worker", "--url", "redis://redis:6379", "cv_training"]
