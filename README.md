# 🩺 Breast Cancer Diagnostic Dashboard (AI-Powered)

This project is a full-stack machine learning application that predicts whether a cell nucleus is **Malignant** (Sick) or **Benign** (Healthy) using 30 real-world biometric measurements.

### 🚀 Features
* **AI Engine:** Built with Scikit-Learn using a **Logistic Regression** model.
* **Accuracy:** Achieved a verified **98.25% accuracy** on the UCI Breast Cancer Wisconsin dataset.
* **Deployment:** Integrated the model into a web dashboard using **Flask (Python)**.
* **Visual Logic:** Includes real-time **S-Curve** probability mapping and a **Confusion Matrix** "Mistake Map."

### 🏛️ Architecture
* **Frontend:** HTML5 with Jinja2 templates for dynamic result rendering.
* **Backend:** Flask REST API serving serialized `.pkl` models.
* **Data Science:** Implemented **StandardScaler** for feature normalization and **Cross-Validation** for reliability.
