# 🧠 Major Project: Brain Tumor Analysis System

This project is a comprehensive MRI-based brain tumor analysis system built with Django and machine learning. It performs tumor detection, segmentation, classification, and stage prediction using deep learning and classical ML models.

## 🔧 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/Manas101010/Major.git
cd Major
```

### 2. Download Required ML Models

Download all the models from the following Google Drive link:

🔗 [Download Models](https://drive.google.com/drive/folders/1XGKj72Jv4iHq1MUYTpu7IG24fJiW2wyy?usp=drive_link)

Place the downloaded models in the following directory:

```
Major/cancer/ml_models/
```

### 3. Install Dependencies

Make sure you have Python 3.8+ and `pip` installed. Then run:

```bash
cd cancer
pip install -r requirements.txt
```

### 4. Run the Server

```bash
python manage.py runserver
```