# 🚀 AI Training System with Docker & K8s-ready Architecture

ระบบ Server สำหรับจัดการการเทรน Model Computer Vision ควบคู่กับ API (รองรับ YOLO, EfficientDet, RT-DETR ฯลฯ)
ระบบนี้ถูกออกแบบเป็น **Long-Running Worker Container พร้อม Message Queue (Redis)** เพื่อยกระดับความสามารถในการสเกล (Scale) และการประมวลผลระยะยาว (Long-Running Tasks) อย่างเป็นระบบ

> **💡 หมายเหตุ:** โค้ดที่เกี่ยวข้องกับฝั่ง Server ทั้งหมด (Backend API, Worker, Redis, Docker Compose) ถูกรวมกันไว้ในโฟลเดอร์ (Repository) นี้แล้ว เพียงแค่โคลน Repo นี้ไปรันที่เซิร์ฟเวอร์ ก็ถือว่าครบจบในที่เดียว

---

## 🏗️ โครงสร้างไฟล์ในโฟลเดอร์นี้ (Architecture)

```text
backend/ (หรือโฟลเดอร์ฝั่ง Server ของคุณ)
├── docker-compose.yml       # ไฟล์หลัก ใช้รันได้ทั้ง Windows (WSL2) และ Ubuntu Server (รองรับ GPU)
├── main.py                  # API กลางที่เขียนด้วย FastAPI (รับ Request, จัดการ Dataset)
├── services/                # โค้ดส่วนบริการรวบรวมภาระงานเข้าสู่คิว (Redis Queue)
├── worker/                  # โค้ดของ Worker (รันค้างไว้ตลอดเพื่อรับงานจาก Queue)
├── dataset/                 # [Bind Mount] ที่เก็บ Dataset (.zip > folder > data.yaml)
└── runs/                    # [Bind Mount] ที่เก็บผลลัพธ์จากการ Train (Model weights, CSV)
```

---

## 🚀 1. คำสั่งรัน Docker

ใช้คำสั่งนี้เพื่อ Build และรันทุกอย่าง (Backend, Redis, Worker, RQ Dashboard) ขึ้นมา

```bash
# พิมพ์คำสั่งในโฟลเดอร์ที่มีไฟล์ docker-compose.yml
cd backend/  # หรือโฟลเดอร์ที่คุณตั้งชื่อไว้

# สำหรับรันแบบทั่วไป (ใช้ได้กับทั้ง Windows WSL ที่อัปเดตแล้ว และ Ubuntu Server)
# หากระบบรองรับและติดตั้งไดรเวอร์ NVIDIA ครบถ้วน Docker จะเชื่อมต่อเข้าการ์ดจอให้เองอัตโนมัติ
docker compose up --build -d
```

> **ถ้ารันแบบดู Log แบบสดๆ** (ไม่รัน Background) ให้ตัด `-d` ออกจากคำสั่งด้านบน

---

## 🔍 2. คำสั่งดู Logs

เพื่อเช็คว่าระบบทำงานปกติไหม หรือดูสถานะว่า Train ไปถึงไหนแล้ว:

```bash
# ดู Log รวมทุก Services (Backend, Redis, Worker)
docker compose logs -f

# 🎯 ดู Log เฉพาะ Worker (สำคัญสุด เอาไว้ดูสถานะโมเดลตอน Train)
docker compose logs -f worker

# ดู Log เฉพาะ Backend API
docker compose logs -f backend

# กด CTRL+C เพื่อออกจากโหมดดู log
```

---

## 3. คำสั่งปิด / Restart Docker

```bash
# ปิดทุก Services (ข้อมูลใน dataset/ และ runs/ จะไม่หาย)
docker compose down

# ตรวจสอบรายชื่อ Container ที่กำลังรัน และ Port ต่างๆ
docker compose ps

# Restart แค่ Worker (สมมติว่ามีการแก้โค้ดใน โฟลเดอร์ worker/)
docker compose restart worker

# Rebuild แค่ Worker (ถ้ามีการแก้ requirements.txt ใน worker/)
docker compose up --build -d worker
```

---

## 🌐 4. หน้าเว็บและพอร์ตที่ใช้งานได้

เมื่อรัน `docker compose up -d` เสร็จแล้ว ระบบและบริการเหล่านี้จะทำงาน (อ้างอิงรหัสไอพีเซิร์ฟเวอร์):

- **Swagger UI (สำหรับเทสต์ระบบ API):** `http://<server-ip>:8000/docs`
- **RQ Dashboard (สำหรับมอนิเตอร์คิว Job):** `http://<server-ip>:9181`

---

## ⚠️ หมวดหมู่ข้อผิดพลาดที่พบบ่อย

1. **OOM (Out Of Memory) หรือ DataLoader Semaphore Exception ของ PyTorch:**
   - สาเหตุลึกๆ เกิดจากขนาดของ Shared Memory (shm) ของ Container ไม่เพียงพอ อาการนี้มักเกิดกับ Dataset ขนาดใหญ่
   - **ทางแก้ไขเบื้องต้น:** ลดค่า `workers` สลับเป็นลดการแบ่ง Batch ตอนคอนฟิกการเทรน. (ในไฟล์ `docker-compose.yml` เราตั้งให้ `worker` ใช้แรมช่วยถึง `8gb` เพื่อแก้ปัญหาชั่วคราวแล้วให้ลองสังเกตดู)

2. **ระบบหา `data.yaml` ไม่เจอตอนสั่งรัน:**
   - มั่นใจว่าไม่ได้เอาโฟลเดอร์ซ้อนโฟลเดอร์จนลึกเกินไปตอนทำการ ZIP ตัว Dataset ไฟล์ `data.yaml` (หรือ `dataset.yaml`) ควรจะอยู่หน้าสุดของ Root Directory เสมอ
