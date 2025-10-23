# Demo Script: Weather Classification MLOps Pipeline

## การนำเสนอ 15 นาที - โครงงานรายวิชา CP413008

### ข้อมูลการนำเสนอ
- **ระยะเวลา:** 15 นาที
- **โครงสร้าง:** 12 นาที Demo + 3 นาที Q&A
- **เป้าหมาย:** แสดงความสามารถของ MLOps Pipeline ที่สมบูรณ์

---

## 🎯 โครงสร้างการนำเสนอ

### 1. Introduction (2 นาที)
### 2. Architecture Overview (2 นาที) 
### 3. Live Demo (8 นาที)
### 4. Results & Conclusion (3 นาที)

---

## 📋 สคริปต์การนำเสนอ

### **[0:00-2:00] Introduction & Project Overview**

**สวัสดีครับ วันนี้ผมจะนำเสนอโครงงาน Weather Classification MLOps Pipeline**

**ปัญหาที่แก้ไข:**
- การจำแนกสภาพอากาศจากภาพถ่ายเป็นงานที่สำคัญในหลายอุตสาหกรรม
- แต่การพัฒนาโมเดล ML เพียงอย่างเดียวไม่เพียงพอสำหรับ Production
- ต้องมี MLOps Pipeline ที่ครบถ้วน

**วัตถุประสงค์:**
1. พัฒนาโมเดล Deep Learning สำหรับจำแนกสภาพอากาศ 5 ประเภท
2. สร้าง MLOps Pipeline ตั้งแต่ Data Validation จนถึง Deployment
3. ใช้ MLflow สำหรับ Experiment Tracking
4. พัฒนา REST API และ CI/CD Pipeline

---

### **[2:00-4:00] Architecture Overview**

**แสดงสไลด์ Architecture Diagram**

**ระบบประกอบด้วย 4 ส่วนหลัก:**

1. **Data Layer**
   - ข้อมูลภาพสภาพอากาศ 18,038 ภาพ
   - Data Validation Script ตรวจสอบคุณภาพ
   - 5 หมวดหมู่: cloudy, foggy, rainy, snowy, sunny

2. **Training Layer**
   - เปรียบเทียบ 3 โมเดล: CNN, MobileNet, EfficientNet
   - MLflow Experiment Tracking
   - Model Registry สำหรับจัดการ versions

3. **Serving Layer**
   - FastAPI REST API
   - Single และ Batch Prediction
   - Dynamic Model Loading

4. **Monitoring Layer**
   - MLflow UI Dashboard
   - Structured Logging
   - Performance Metrics

---

### **[4:00-12:00] Live Demo**

#### **Demo 1: Data Validation (1.5 นาที)**

```bash
# เปิด Terminal 1
cd C:\projectML\weather_classification_mlops\scripts
python 01_data_validation.py ../../data ../artifacts
```

**อธิบายขณะรัน:**
- "ตอนนี้กำลังตรวจสอบคุณภาพข้อมูล 18,038 ภาพ"
- "ระบบตรวจสอบภาพเสียหาย, ขนาดไม่เหมาะสม, และ class imbalance"

**แสดงผลลัพธ์:**
```
Total Images: 18,038
Valid Images: 18,029
Corrupted Images: 4
Invalid Size Images: 5
Dataset Imbalanced: True (ratio: 5.32:1)
```

#### **Demo 2: MLflow Experiment Tracking (2 นาที)**

**เปิด Browser ไป MLflow UI:**
```
http://127.0.0.1:5000
```

**แสดงและอธิบาย:**
1. **Experiments List:** แสดง weather_classification experiments
2. **Run Comparison:** เปรียบเทียบ CNN, MobileNet, EfficientNet
3. **Metrics Visualization:** Accuracy, Loss curves
4. **Model Artifacts:** แสดง saved models และ parameters

**Key Points:**
- "EfficientNet ให้ accuracy สูงสุด 87.89%"
- "MobileNet มี balance ดีระหว่าง accuracy และ speed"
- "ทุก experiment ถูกบันทึกอย่างเป็นระบบ"

#### **Demo 3: Model Registry (1 นาที)**

**ใน MLflow UI:**
1. ไปที่ Models tab
2. แสดง registered models
3. อธิบาย model versioning และ staging

**อธิบาย:**
- "Model Registry ช่วยจัดการ model versions"
- "สามารถ promote model จาก Staging ไป Production"
- "Track model lineage และ metadata"

#### **Demo 4: API Service (3 นาที)**

**ตรวจสอบ API Status:**
```bash
# เปิด Browser
http://127.0.0.1:8000/docs
```

**แสดง FastAPI Swagger UI:**
1. **Health Check Endpoint**
2. **Model Info Endpoint** 
3. **Available Models**

**Demo Single Image Prediction:**
```bash
# เปิด Terminal 2
cd C:\projectML\weather_classification_mlops\scripts
python test_with_sample_image.py
```

**อธิบายขณะรัน:**
- "ทดสอบการทำนายภาพเดี่ยว 5 ภาพ"
- "แต่ละภาพใช้เวลาประมาณ 245ms"
- "ระบบให้ confidence score สำหรับแต่ละการทำนาย"

**แสดงผลลัพธ์:**
```
✅ cloudy.jpg -> Predicted: cloudy (98.5%)
✅ sunny.jpg -> Predicted: sunny (96.2%)
⚠️ foggy.jpg -> Predicted: cloudy (78.3%) [Expected: foggy]
✅ snowy.jpg -> Predicted: snowy (94.7%)
⚠️ rainy.jpg -> Predicted: cloudy (82.1%) [Expected: rainy]
```

**Demo Batch Prediction:**
- "ทดสอบ batch prediction 5 ภาพพร้อมกัน"
- "ใช้เวลาประมาณ 890ms สำหรับ 5 ภาพ"
- "Efficient สำหรับการประมวลผลจำนวนมาก"

#### **Demo 5: CI/CD Pipeline (0.5 นาที)**

**แสดงใน GitHub:**
1. เปิด `.github/workflows/ci-cd.yml`
2. อธิบาย pipeline stages:
   - Code Quality (Linting, Security)
   - Unit Tests
   - Data Validation
   - Model Training Test
   - API Testing
   - Security Scanning

**Key Points:**
- "Automated testing ทุกครั้งที่ push code"
- "Multi-stage pipeline รับประกันคุณภาพ"
- "Security scanning ด้วย Trivy"

---

### **[12:00-15:00] Results & Conclusion**

#### **ผลการดำเนินงาน (1.5 นาที)**

**Model Performance:**
| Model | Accuracy | Training Time |
|-------|----------|---------------|
| CNN | 82.34% | 45 min |
| MobileNetV2 | 85.67% | 32 min |
| EfficientNet | **87.89%** | 58 min |

**System Performance:**
- API Response Time: 245ms (average)
- Batch Processing: 890ms for 5 images
- Data Validation: 18,029/18,038 valid images
- CI/CD Pipeline: 95%+ success rate

#### **ความท้าทายและการแก้ไข (1 นาที)**

1. **Class Imbalance:** 
   - ปัญหา: Foggy (7%) vs Cloudy (37.2%)
   - แก้ไข: Stratified sampling + Data augmentation

2. **Model Loading Issues:**
   - ปัญหา: TensorFlow naming conflicts
   - แก้ไข: Function aliasing

3. **API Response Format:**
   - ปัญหา: Test script compatibility
   - แก้ไข: Standardized response format

#### **สรุปและการประยุกต์ใช้ (0.5 นาที)**

**ความสำเร็จ:**
- ✅ MLOps Pipeline ครบถ้วน (20/20 คะแนน)
- ✅ Production-ready API
- ✅ Comprehensive monitoring
- ✅ Automated CI/CD

**การประยุกต์ใช้:**
- Weather monitoring systems
- Agricultural applications  
- Transportation safety
- Smart city infrastructure

---

## 🛠️ เตรียมความพร้อมก่อน Demo

### **Pre-Demo Checklist (30 นาที ก่อนนำเสนอ)**

#### **1. ตรวจสอบ Services (5 นาที)**
```bash
# Terminal 1: MLflow UI
cd C:\projectML\weather_classification_mlops\scripts
mlflow ui --host 127.0.0.1 --port 5000

# Terminal 2: API Service  
cd C:\projectML\weather_classification_mlops\scripts
uvicorn 04_load_and_predict:app --host 127.0.0.1 --port 8000 --reload

# Terminal 3: Test Ready
cd C:\projectML\weather_classification_mlops\scripts
```

#### **2. เตรียม Browser Tabs (2 นาที)**
- Tab 1: http://127.0.0.1:5000 (MLflow UI)
- Tab 2: http://127.0.0.1:8000/docs (FastAPI Swagger)
- Tab 3: GitHub Repository (.github/workflows/ci-cd.yml)
- Tab 4: Project Report (PROJECT_REPORT.md)

#### **3. เตรียมไฟล์ทดสอบ (3 นาที)**
```bash
# ตรวจสอบไฟล์ sample images
ls C:\projectML\data\cloudy\cloudy1.jpg
ls C:\projectML\data\sunny\sunny1.jpg
ls C:\projectML\data\foggy\foggy1.jpg
ls C:\projectML\data\rainy\rainy1.jpg
ls C:\projectML\data\snowy\snowy1.jpg
```

#### **4. ทดสอบ Commands (10 นาที)**
```bash
# Test 1: Data Validation
python 01_data_validation.py ../../data ../artifacts

# Test 2: API Testing
python test_with_sample_image.py

# Test 3: Health Check
curl http://127.0.0.1:8000/health
```

#### **5. เตรียมสไลด์และสคริปต์ (10 นาที)**
- พิมพ์สคริปต์นี้ออกมา
- เตรียม Architecture diagram
- ทบทวนจุดสำคัญที่ต้องเน้น

---

## 🎤 Tips สำหรับการนำเสนอ

### **การพูด**
1. **พูดช้าและชัดเจน** - ให้เวลาผู้ฟังทำความเข้าใจ
2. **อธิบายขณะ Demo** - ไม่ให้เกิดช่วงเงียบ
3. **เน้นจุดเด่น** - MLOps completeness, Production readiness
4. **เตรียมรับมือ Error** - มี backup plan หาก demo ล้มเหลว

### **การจัดการเวลา**
- **ใช้ Timer** - ตั้งเตือนทุก 3 นาที
- **มี Buffer Time** - เสร็จก่อนเวลา 1-2 นาที
- **Skip ได้** - หาก demo ช้า ให้ข้าม details บางส่วน

### **การรับมือปัญหา**
1. **API ไม่ตอบสนอง:** แสดง screenshots ที่เตรียมไว้
2. **MLflow UI ช้า:** อธิบายจาก screenshots
3. **Network Issues:** ใช้ offline demo หรือ recorded video

### **Q&A Preparation**
**คำถามที่อาจถูกถาม:**
1. **"ทำไมเลือก EfficientNet?"** 
   - Balance ระหว่าง accuracy และ efficiency
   - State-of-the-art architecture
   
2. **"จัดการ class imbalance อย่างไร?"**
   - Data augmentation สำหรับ minority classes
   - Stratified sampling
   - Weighted loss function

3. **"ระบบรองรับ scale ได้แค่ไหน?"**
   - Current: Single instance
   - Future: Kubernetes, Load balancing
   - Horizontal scaling ready

4. **"Security measures มีอะไรบ้าง?"**
   - Input validation
   - Dependency scanning
   - Container security

---

## 📊 Backup Materials

### **Screenshots ที่ควรเตรียม**
1. MLflow Experiments comparison
2. Model Registry interface  
3. API Swagger documentation
4. GitHub Actions pipeline
5. Data validation results

### **Demo Video (Optional)**
- บันทึก demo ไว้ 5 นาที
- ใช้ในกรณีที่ live demo มีปัญหา

### **Key Statistics ที่ต้องจำ**
- **Dataset:** 18,038 images, 5 classes
- **Best Model:** EfficientNet 87.89% accuracy
- **API Performance:** 245ms average response
- **Data Quality:** 18,029/18,038 valid images
- **Pipeline Success:** 95%+ CI/CD success rate

---

## ✅ Final Checklist

**1 ชั่วโมงก่อนนำเสนอ:**
- [ ] ทุก services ทำงานปกติ
- [ ] Browser tabs เตรียมพร้อม
- [ ] Test commands ทำงานได้
- [ ] Backup materials พร้อม
- [ ] Timer และ notes เตรียมพร้อม

**15 นาทีก่อนนำเสนอ:**
- [ ] Final test ทุก endpoints
- [ ] ปิด applications ที่ไม่จำเป็น
- [ ] เตรียม mindset และความมั่นใจ

**ขณะนำเสนอ:**
- [ ] เริ่มด้วยความมั่นใจ
- [ ] ติดตาม timer
- [ ] มีปฏิสัมพันธ์กับผู้ฟัง
- [ ] สรุปจุดสำคัญในตอนท้าย

---

**Good Luck! 🚀**

*"แสดงให้เห็นว่าเราไม่ได้แค่สร้างโมเดล แต่สร้างระบบที่พร้อมใช้งานจริง"*