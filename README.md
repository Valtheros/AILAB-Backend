# 🚀 YOLO Training System Backend

ระบบ Backend สำหรับจัดการการเทรน Model YOLO v11 ผ่าน API โดยรอบรับการทำงานด้วย Docker และ GPU Acceleration (NVIDIA) พัฒนาด้วย FastAPI

## ✨ Features

- **FastAPI Integration**: รวดเร็วและใช้งานง่ายด้วย Swagger UI
- **Dockerized Training**: รันการเทรนบน Docker Container (Ultraalytics image) เพื่อความเสถียรและความง่ายในการติดตั้ง
- **GPU Acceleration**: รองรับการใช้งาน GPU ผ่าน Docker
- **Real-time Monitoring**: สามารถดึง Status และ Logs ข้อมูลการเทรนได้ตลอดเวลา
- **Auto Dataset Config**: จัดการและค้นหา `data.yaml` อัตโนมัติ พร้อมปรับแต่ง path ให้เหมาะสมกับ Docker environment

## 🛠 Tech Stack

- **Laguage**: Python 3.x
- **Framework**: [FastAPI](https://fastapi.tiangolo.com/)
- **Containerization**: [Docker](https://www.docker.com/)
- **YOLO Framework**: [Ultralytics YOLO v11](https://github.com/ultralytics/ultralytics)

## 🏗 Prerequisites

1.  **Python 3.10+**
2.  **Docker Desktop** (สำหรับ Windows) หรือ **Docker Engine** (สำหรับ Linux)
3.  **NVIDIA Docker Runtime** (หากต้องการใช้ GPU สำหรับการเทรน)
4.  **NVIDIA Drivers** ติดตั้งบนเครื่อง Host

## 📥 Installation & Setup

1. **Clone project** (ถ้ายังไม่ได้ทำ)
2. **ติดตั้ง dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

   _หมายเหตุ: ตรวจสอบว่ามี Docker SDK for Python (`docker`) และ `fastapi`, `uvicorn`, `pyyaml` ติดตั้งอยู่_

3. **เตรียม Dataset**:
   วางโฟลเดอร์ Dataset ไว้ในโฟลเดอร์ `dataset/` โดยภายในต้องมีไฟล์ `data.yaml`

4. **เริ่มรัน Server**:
   ```bash
   python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

## 📡 API Endpoints

### 🟢 1. Start Training

**Endpoint**: `POST /api/train`
**Request Body**:

```json
{
  "model_size": "n",
  "epochs": 100,
  "batch_size": 16,
  "project_name": "my_yolo_train"
}
```

_`model_size`: เลือกได้จาก n, s, m, l, x (เช่น n สำหรับ yolov11n.pt)_

### 🔵 2. Check Status

**Endpoint**: `GET /api/status/{container_id}`
ดึงสถานะปัจจุบันของ Container (running, exited, etc.)

### 🟡 3. Get Logs

**Endpoint**: `GET /api/logs/{container_id}`
ดึง Logs ล่าสุดจากการเทรนแบบ Real-time

## 📂 Project Structure

```bash
backend/
├── dataset/             # เก็บข้อมูลรูปภาพและ data.yaml สำหรับเทรน
├── runs/                # เก็บผลลัพธ์การเทรน (weights, plots)
├── services/
│   └── training_service.py # Logic การจัดการ Docker container
├── main.py              # FastAPI Main Application
└── README.md            # เอกสารประกอบการใช้งาน
```

## ⚠️ Notes

- ข้อมูลการเทรนจะถูกเก็บไว้ที่โฟลเดอร์ `runs/` เมื่อเสร็จสิ้น
- ตรวจสอบให้แน่ใจว่า Docker Desktop กำลังรันอยู่ก่อนเริ่มเทรน
- ระบบตั้งค่า `shm_size="8g"` เพื่อป้องกันหน่วยความจำแชร์ไม่พอขณะเทรน

---

