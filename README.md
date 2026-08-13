# Force-Analysis

# 🏋️‍♂️ Free JumpAnz Team - Biomechanics Analysis System

ระบบวิเคราะห์กราฟแรงและตัวแปรทางชีวกลศาสตร์สำหรับการทดสอบ **Countermovement Jump (CMJ)** พัฒนาโดย **PRIMA MOTION TECHNOLOGY** 

แอปพลิเคชันนี้ออกแบบมาเพื่อรองรับการนำเข้าไฟล์ข้อมูลแรงดิบจากแบรนด์ชั้นนำ หลากหลายฟอร์แมต ประมวลผลเฟสการเคลื่อนไหวอัตโนมัติ คำนวณตัวแปรมาตรฐานสากล (*Anicic et al., 2023*) และออกรายงานสรุปรูปแบบ A4 PDF

---

## 🌟 Key Features

* **Multi-Format Data Importer:**
  * **MuscleLab CSV (.csv)** — รองรับการตรวจจับและตัดเฟสหลุดแผ่นข้างเดียว (Trailing Single-Plate Truncation) อัตโนมัติที่ $1000\text{ Hz}$
  * **VALD ForceDecks (CSV/TSV)** — รองรับทั้งระบบ Multi-column และ Single-time
  * **Qualisys QTM JSON (.json)** — เลือกจับคู่ Force Plate แบบยืดหยุ่นพร้อมฟังก์ชันสลับฝั่ง (L/R Swap)
  * **Single CSV (C-Force)** & **Dual TSV (Plate A + B)**
* **Signal Conditioning & Filtering Engine:**
  * **Zero-Phase 4th-Order Low-Pass Butterworth Filter** (พร้อมระบบจัดการ Edge Transients)
  * **Moving Average Filter**
  * **Raw Data Bypass**
* **Interactive Phase Control & Biomechanical Pictograms:**
  * ตรวจจับเฟสการกระโดดอัตโนมัติ ($Unweighting \rightarrow Braking \rightarrow Propulsive \rightarrow Flight \rightarrow Landing$)
  * ควบคุมและปรับแต่งขอบเขตเฟสผ่านสไลเดอร์และกล่องป้อนตัวเลขแบบล็อกลำดับ ($1000\text{ Hz}$ Precision)
  * แสดงรูปภาพประกอบท่าทางชีวกลศาสตร์ (Pictograms) บนแกนเวลา
* **L/R Asymmetry Analysis:**
  * กราฟวิเคราะห์ความไม่สมดุลซ้าย-ขวาพร้อมแถบแจ้งเตือน Threshold Alert (%)
  * ระบุฝั่งเด่นชัดเจน (**Left Dominant** / **Right Dominant**)
* **Automated PDF Reporting:**
  * ส่งออกรายงาน A4 PDF สรุปกราฟ $F_z$, กราฟ Asymmetry %, รูป Pictograms และตารางตัวแปร 4 หมวดหลัก

---

## 📊 Biomechanical Metrics Classification (Anicic et al., 2023)

ระบบคำนวณและจำแนกตัวแปรออกเป็น 4 หมวดหมู่หลักตามงานวิจัย:

1. **Performance Component (59% Variance):** Jump Height (Flight Time & Impulse-Momentum), Take-off Velocity, RSI Modified, Landing Impulse, Propulsive Power & Impulse
2. **Eccentric Component (16% Variance):** Mean Braking Force, Braking Impulse, Eccentric Braking RFD, Unloading Impulse, Peak Negative Velocity
3. **Concentric Component (11% Variance):** Peak/Mean Propulsive Force, Peak Braking Force, Time to Peak Force, Concentric Impulse (P1: 0-50ms, P2: 50-100ms)
4. **Jump Strategy Component (6% Variance):** Propulsive Duration, Countermovement COM Depth, Leg Stiffness, Flight Time to Jump Time Ratio

---
