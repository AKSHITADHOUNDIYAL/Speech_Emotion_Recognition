# 🎙️ Speech Emotion Recognition Web App

This project combines a powerful machine learning model with a modern web interface to detect human emotions from **speech/audio input**.

---

## 🧠 Machine Learning Model

At the core of this system is a **fine-tuned Wav2Vec2 model**, based on the `facebook/wav2vec2-large-xlsr-53` architecture. It has been trained on the **TESS Toronto Emotional Speech Dataset** with 7 emotion classes:

> 😃 Happy | 😡 Angry | 😢 Sad | 😱 Fear | 😖 Disgust | 😲 Surprise | 😐 Neutral

The model includes:
- **Wav2Vec2 Encoder** for feature extraction
- **Custom classification head** (PyTorch)
- **Preprocessing** with noise augmentation & normalization
- Exported and used via `emotion_model.pth`

---

## 🌐 Full-Stack Web Application

The application consists of a **FastAPI backend** and a **React.js frontend**, designed for an interactive and real-time emotion recognition experience.

### 🧠 Machine Learning
- `transformers==4.37.2`
- `peft==0.3.0`
- `accelerate==0.25.0`
- `torch`, `torchaudio`
- `scikit-learn`, `librosa`

### 🖥️ Backend
- `FastAPI` - Lightweight and fast backend framework
- `Uvicorn` - ASGI server
- `Pydantic` - Data validation

### 🌐 Frontend
- `React.js` (Latest version)
- `Tailwind CSS` & `Framer Motion` for beautiful animations and transitions
- `axios` for HTTP requests
- `MediaRecorder` API to capture live audio

---

## 📁 Project Structure
`Speech-Emotion-Recognition/
├── Backend/
│   ├── main.py                
│   ├── emotion_detection.py   
│   ├── emotion_model.pth      
│   └── Extraction
|        └── emotion_recognizer.py 
|   └── audio_processor.py
|    └── Prediction 
|       └── model
|            └── emotion_detection.plt 
|        
│
├── Frontend/
│   ├── src/
│   │   ├── components/
│   │   │   |            
│   │   │   ├── ResultDashboard.jsx
│   │   │   ├── SpectrogramPlot.jsx     
│   │   │   ├── EmotionBarChart.jsx     
│   │   │   └── WaveformPlot.jsx      
│   │   └── index.js
        ├── App.jsx        
│   ├── public/
│   └── package.json              
│
└── README.md`

## ⚙️ Features

- 🎧 Upload `.wav` audio files
- 🎙️ Record live audio using mic
- 📈 Real-time emotion prediction
- 🧠 Visual feedback via animated emotion bars
- 🚀 Fast and responsive using modern frontend stack

---

## 🛠️ Tech Stack

- **Backend:** FastAPI, PyTorch, Transformers (Hugging Face), Wav2Vec2
- **Frontend:** React.js, Tailwind CSS, Framer Motion
- **Model Training:** Wav2Vec2 fine-tuned with `Trainer` API from Hugging Face

---

## 🚀 How to Run

1. **Clone the repo:**
   ```bash
   git clone https://github.com/your-username/Speech-Emotion-Recognition.git
   cd Speech-Emotion-Recognition

2. **  Frontend setup (inside Frontend/) **
   ``` bash
    cd Frontend
    npm install
    npm start

3. **Backend setup (inside Backend/)**
    ```bash
      cd Backend
      pip install -r requirements.txt
      uvicorn main:app --reload

## Google Drive Folder

You can access the necessary files from our [Google Drive Folder](https://drive.google.com/drive/folders/1qPnUan8NRQEQxBT2IbY5NXk-7sxnIUbG?usp=drive_link).
This contains emotion_detection.plt
