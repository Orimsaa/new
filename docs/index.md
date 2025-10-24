---
layout: default
title: Weather Classification MLOps
---

# Weather Classification MLOps 🌤️

ยินดีต้อนรับสู่หน้าโครงการ! เว็บไซต์นี้เผยแพร่ผ่าน GitHub Pages จากโฟลเดอร์ `/docs` บนสาขา `master` ของโปรเจค

## เกี่ยวกับโปรเจค
ระบบ MLOps สำหรับจำแนกสภาพอากาศจากภาพถ่าย รองรับ 5 หมวดหมู่: Sunny, Cloudy, Rainy, Snowy, Foggy พร้อม MLflow, CI/CD และ API ด้วย FastAPI

## เริ่มต้นใช้งาน
- โค้ดบน GitHub: [Repository](https://github.com/Orimsaa/ml_projectV1)
- Clone และติดตั้ง:
```bash
git clone https://github.com/Orimsaa/ml_projectV1.git
cd weather_classification_mlops
pip install -r requirements.txt
```

## รัน API แบบท้องถิ่น
```bash
python scripts/04_load_and_predict.py --host 0.0.0.0 --port 8000
```
เปิดเบราว์เซอร์ที่ `http://localhost:8000` และลอง `GET /health`

## MLflow Tracking
```bash
mlflow ui
```
แล้วเข้า `http://localhost:5000`

## เอกสารที่เกี่ยวข้อง
- README: [เปิดดู](../README.md)
- Demo Script: [เปิดดู](../DEMO_SCRIPT.md)
- Presentation Slides: [เปิดดู](../PRESENTATION_SLIDES.md)
- Project Report: [เปิดดู](../PROJECT_REPORT.md)

## Docker
Image ถูก build/push ไปที่ GitHub Container Registry:
```
ghcr.io/Orimsaa/weather-classification-mlops:latest
```
รันด้วย:
```bash
docker run -p 8000:8000 ghcr.io/Orimsaa/weather-classification-mlops:latest
```

---
อัพเดตล่าสุด: {{ site.time | date: '%Y-%m-%d' }}