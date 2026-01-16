# AgriVision: See & Spray Prototype 🚜

**An end-to-end Computer Vision application for precision agriculture.**

This project simulates a "Green-on-Green" weed detection system (See & Spray) using **YOLOv8**. It detects broadleaf weeds in crop rows and calculates the economic impact of precision spraying versus broadcast spraying.

## 🎯 Key Features
* **Custom Weed Detection:** Trained YOLOv8n model to identify weeds vs. crops.
* **Economic Impact Dashboard:** Calculates herbicide cost savings per acre in real-time.
* **Edge-Ready Logic:** Optimized inference pipeline suitable for field deployment.

## 🛠️ Tech Stack
* **Model:** YOLOv8 (Ultralytics)
* **App:** Streamlit
* **Language:** Python 3.10
* **Data:** Custom annotated agricultural dataset

## 🚀 How to Run
```bash
pip install ultralytics streamlit
streamlit run app.py
```

---
*Built by Sushma Mutyala (Ag Officer turned Data Scientist)*
