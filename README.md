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

---

## 🌐 Full-Stack Web Application
Web-application based on ML model for recognition of emotion for selected audio file using Streamlit.

### 🧠 Machine Learning
- `transformers==4.37.2`
- `peft==0.3.0`
- `accelerate==0.25.0`
- `torch`, `torchaudio`
- `scikit-learn`, `librosa`

### 🌐 Frontend using Streamlit
- `Streamlit` for building the web interface
- `streamlit_audio_recorder` for using live mic 
- `Matplotlib` and `Librosa` for waveform and spectrogram visualizations
- Custom UI built using native Streamlit components

---

## ⚙️ Features

- 🎧 Upload `.wav` audio files
- 🎙️ Record live audio using mic
- 📈 Real-time emotion prediction
- 🧠 Visual feedback with waveform and spectrogram plots
- 🌟 Simple and elegant UI using Streamlit components and matplotlib visualizations

---

## 🛠️ Tech Stack

- **Frontend & UI:**  Streamlit
- **Model Training:** Wav2Vec2 fine-tuned with `Trainer` API from Hugging Face

---

## 🚀 How to Run

1. **Clone the repo:**
   ```bash
   git clone https://github.com/AKSHITADHOUNDIYAL/Speech_Emotion_Recognition/.git
   cd Speech-Emotion-Recognition

2. **  Install dependencies: **
   ``` bash
    pip install -r requirements.txt

3. **Run the Streamlit app:**
    ```bash
    streamlit run app.py
