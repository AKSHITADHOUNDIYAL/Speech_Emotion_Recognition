import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import os
import sounddevice as sd
import soundfile as sf
import tempfile
from tensorflow.keras.models import load_model
from melspec import plot_colored_polar, get_melspec, get_title
import requests
from streamlit_lottie import st_lottie
import time

# -------------------- Setup --------------------
st.set_page_config(
    page_title="🎭 Speech Emotion Recognition", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load Lottie animations
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Load model
model = load_model("model3.h5")

EMOTIONS = ['fear', 'disgust', 'neutral', 'happy', 'sad', 'surprise', 'angry']
CAT3 = ["positive", "neutral", "negative"]

COLOR_DICT = {
    "neutral": "grey",
    "positive": "green",
    "happy": "green",
    "surprise": "orange",
    "fear": "purple",
    "negative": "red",
    "angry": "red",
    "sad": "lightblue",
    "disgust": "brown"
}

EMOTION_GIF = {
    "happy": "https://media.giphy.com/media/3o7abAHdYvZdBNnGZq/giphy.gif",  
    "sad": "https://media.giphy.com/media/L95W4wv8nnb9K/giphy.gif",  
    "angry": "https://media.giphy.com/media/l3V0j3ytFyGHqiV7W/giphy.gif",  
    "fear": "https://media.giphy.com/media/26ufdipQqU2lhNA4g/giphy.gif",  
    "surprise": "https://media.giphy.com/media/3o7TKsQ8gqVrXhq5Li/giphy.gif",  
    "neutral": "https://media.giphy.com/media/3o7TKsQ8gqVrXhq5Li/giphy.gif",  
    "disgust": "https://media.giphy.com/media/3o7TKsQ8gqVrXhq5Li/giphy.gif"  
}

st.image("images/schema.png", use_container_width=True, caption="System Overview")

# -------------------- Enhanced UI Styling --------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');

:root {
    --primary: #6C63FF;
    --secondary: #4D44DB;
    --light: #F8F9FA;
    --dark: #212529;
    --success: #28A745;
    --danger: #DC3545;
    --warning: #FFC107;
    --info: #17A2B8;
}

* {
    font-family: 'Poppins', sans-serif;
}

body {
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
}

.stApp {
    background: transparent;
}

.stButton>button {
    border: 2px solid var(--primary);
    border-radius: 20px;
    color: white;
    background: var(--primary);
    padding: 0.5rem 1.5rem;
    transition: all 0.3s ease;
}

.stButton>button:hover {
    background: var(--secondary);
    color: white;
    border-color: var(--secondary);
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
}

.stRadio>div {
    flex-direction: row;
    align-items: center;
    gap: 10px;
}

.stRadio>div>label {
    margin-bottom: 0;
    padding: 8px 15px;
    border-radius: 20px;
    transition: all 0.3s ease;
}

.stRadio>div>label:hover {
    background: rgba(108, 99, 255, 0.1);
}

.stRadio>div>div:first-child {
    display: none;
}

.stFileUploader>div>div>button {
    border-radius: 20px;
}

.custom-card {
    border-radius: 15px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    padding: 1.5rem;
    background: white;
    transition: transform 0.3s ease, box-shadow 0.3s ease;
    margin-bottom: 1.5rem;
}

.custom-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 8px 25px rgba(0,0,0,0.15);
}

.traffic-light {
    display: flex;
    justify-content: center;
    align-items: center;
    margin: 20px 0;
    gap: 15px;
}

.light {
    width: 40px;
    height: 40px;
    border-radius: 50%;
    box-shadow: 0 0 10px rgba(0,0,0,0.2);
    transition: all 0.3s ease;
    position: relative;
}

.light::after {
    content: '';
    position: absolute;
    top: -5px;
    left: -5px;
    right: -5px;
    bottom: -5px;
    border-radius: 50%;
    opacity: 0;
    box-shadow: 0 0 20px currentColor;
    transition: opacity 0.3s ease;
}

.light.red {
    background-color: #FF5E5E;
}

.light.red.active {
    background-color: #FF0000;
    box-shadow: 0 0 15px #FF0000;
}

.light.red.active::after {
    opacity: 0.7;
}

.light.yellow {
    background-color: #FFD166;
}

.light.yellow.active {
    background-color: #FFC107;
    box-shadow: 0 0 15px #FFC107;
}

.light.yellow.active::after {
    opacity: 0.7;
}

.light.green {
    background-color: #8AC926;
}

.light.green.active {
    background-color: #28A745;
    box-shadow: 0 0 15px #28A745;
}

.light.green.active::after {
    opacity: 0.7;
}

@keyframes pulse {
    0% { transform: scale(1); }
    50% { transform: scale(1.05); }
    100% { transform: scale(1); }
}

.pulse {
    animation: pulse 2s infinite;
}

.emoji-header {
    font-size: 2.5rem;
    margin-bottom: 0.5rem;
}

.progress-container {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 10px;
}

.progress-label {
    min-width: 80px;
    font-weight: 600;
}

.progress-percent {
    min-width: 60px;
    text-align: right;
    font-weight: 600;
}

</style>
""", unsafe_allow_html=True)

# -------------------- Sidebar --------------------
with st.sidebar:
    st.markdown("<h1 style='text-align: center; color: #6C63FF;'>🎤 Emotion Detector</h1>", unsafe_allow_html=True)
    
    # Animation in sidebar
    lottie_mic = load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_pojzngga.json")
    st_lottie(lottie_mic, height=150, key="sidebar_mic")
    
    st.sidebar.image("images/smileyfacesboxes.jpg", use_container_width=True)
    st.sidebar.image("images/emotion3.jpg", use_container_width=True)
    
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center;">
        <p>Analyze speech emotions in real-time</p>
        <p>Upload or record audio to get started</p>
    </div>
    """, unsafe_allow_html=True)

# -------------------- Main Content --------------------
st.title("🎭 Speech Emotion Recognition")
st.markdown("""
<div class="custom-card">
    <h3 style="color: #6C63FF;">Discover Emotions in Speech</h3>
    <p>This application uses advanced machine learning to detect emotions from audio recordings. 
    Upload an audio file or record directly to analyze the emotional content.</p>
</div>
""", unsafe_allow_html=True)

# -------------------- Functions --------------------
def extract_mfccs(path, limit):
    y, sr = librosa.load(path, sr=None)
    mfcc_features = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    padded = np.zeros((40, limit))
    padded[:, :min(mfcc_features.shape[1], limit)] = mfcc_features[:, :limit]
    return padded, sr

def analyze_emotions(path):
    mfccs, sr = extract_mfccs(path, model.input_shape[-1])
    mfccs = mfccs.reshape(1, *mfccs.shape)
    pred = model.predict(mfccs)[0]
    if len(pred) < 7:
        pred = np.pad(pred, (0, 7 - len(pred)), mode='constant')
    pos = pred[3] + pred[5] * .5
    neu = pred[2] + pred[5] * .5 + pred[4] * .5
    neg = pred[0] + pred[1] + pred[4] * .5
    data3 = np.array([pos, neu, neg])
    return pred, data3, sr

def detect_gender(path):
    y, sr = librosa.load(path)
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch = pitches[magnitudes > np.median(magnitudes)].mean()
    gender = "male" if pitch < 160 else "female"
    return gender

def show_emotion_gif(emotion):
    gif_url = EMOTION_GIF.get(emotion, "")
    if gif_url:
        st.image(gif_url, width=300)

def traffic_light(color):
    html = f"""
    <div class="traffic-light">
        <div class="light red {'active' if color == 'red' else ''}"></div>
        <div class="light yellow {'active' if color == 'yellow' else ''}"></div>
        <div class="light green {'active' if color == 'green' else ''}"></div>
    </div>
    """
    return html

def display_results(path, pred, data3, sr):
    wav, _ = librosa.load(path, sr=sr)
    
    with st.spinner("Analyzing results..."):
        with st.container():
            st.markdown("## 📊 Analysis Results")
            
            # Emotion detection card with animation
            main_emotion = EMOTIONS[np.argmax(pred)]
            emoji = {
                "happy": "😊", 
                "sad": "😢", 
                "angry": "😠", 
                "fear": "😨", 
                "surprise": "😲", 
                "neutral": "😐", 
                "disgust": "🤢"
            }.get(main_emotion, "🎭")
            
            st.markdown(f"""
            <div class="custom-card pulse">
                <h2 style="color: {COLOR_DICT.get(main_emotion, '#6C63FF')}; text-align: center;">
                    {emoji} Detected Emotion: <strong>{main_emotion.capitalize()}</strong> {emoji}
                </h2>
            </div>
            """, unsafe_allow_html=True)
            
            show_emotion_gif(main_emotion)
            
            # Audio visualization in tabs
            tab1, tab2, tab3 = st.tabs(["📈 Waveform", "🎛️ MFCC", "🌈 Spectrogram"])
            
            with tab1:
                fig = plt.figure(figsize=(10, 4))
                librosa.display.waveshow(wav, sr=sr, color='#6C63FF')
                plt.title("Waveform", fontsize=14)
                plt.grid(True, alpha=0.3)
                st.pyplot(fig)
            
            with tab2:
                mfcc_feat = librosa.feature.mfcc(y=wav, sr=sr, n_mfcc=40)
                fig = plt.figure(figsize=(10, 4))
                librosa.display.specshow(mfcc_feat, sr=sr, x_axis='time', cmap='viridis')
                plt.colorbar(format='%+2.0f dB')
                plt.title("MFCC", fontsize=14)
                st.pyplot(fig)
            
            with tab3:
                stft = np.abs(librosa.stft(wav))
                spec_db = librosa.amplitude_to_db(stft, ref=np.max)
                fig = plt.figure(figsize=(10, 4))
                librosa.display.specshow(spec_db, sr=sr, x_axis='time', y_axis='log', cmap='magma')
                plt.colorbar(format='%+2.0f dB')
                plt.title("Spectrogram (dB)", fontsize=14)
                st.pyplot(fig)
            
            # Emotion charts in columns
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 3-Class Analysis")
                title3 = get_title(data3, categories=CAT3, first_line="Emotion Categories")
                fig = plt.figure(figsize=(6, 6))
                plot_colored_polar(fig, predictions=data3, categories=CAT3, title=title3)
                st.pyplot(fig)
                
                fig = plt.figure(figsize=(6, 4))
                bars = plt.bar(CAT3, data3, color=[COLOR_DICT[c] for c in CAT3])
                plt.ylim(0, 1)
                plt.title("3-Class Probabilities", fontsize=14)
                plt.grid(True, alpha=0.3)
                for bar in bars:
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height,
                             f'{height:.2f}',
                             ha='center', va='bottom')
                st.pyplot(fig)
            
            with col2:
                st.markdown("### 7-Class Analysis")
                title7 = get_title(pred, categories=EMOTIONS, first_line="Detailed Emotions")
                fig = plt.figure(figsize=(6, 6))
                plot_colored_polar(fig, predictions=pred, categories=EMOTIONS, title=title7)
                st.pyplot(fig)
                
                fig = plt.figure(figsize=(8, 4))
                pred = np.array(pred).flatten()
                if len(pred) != len(EMOTIONS):
                    pred = np.pad(pred, (0, len(EMOTIONS) - len(pred)), mode='constant')
                bars = plt.bar(EMOTIONS, pred, color=[COLOR_DICT[e] for e in EMOTIONS])
                plt.xticks(rotation=45, ha='right')
                plt.ylim(0, 1)
                plt.title("7-Class Probabilities", fontsize=14)
                plt.grid(True, alpha=0.3)
                for bar in bars:
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height,
                             f'{height:.2f}',
                             ha='center', va='bottom', rotation=90)
                st.pyplot(fig)
            
            
            st.markdown("#### Gender")

            gender_col, emo_col = st.columns([1, 3])
            gender = detect_gender(path)

            with gender_col:
                if gender == "male":
                    st.image("https://media1.giphy.com/media/v1.Y2lkPTc5MGI3NjExN3A5Y2dyM2hwZWszbmhieTQyOXFnZHVkZ2RiNm42YjVraWpxbG81YSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/zKIFCGg9aV3dm/giphy.gif", width=150)  
                    st.write("Male")
                elif gender == "female":
                    st.image("https://media0.giphy.com/media/v1.Y2lkPTc5MGI3NjExN2x1bmJ4bm42YXY3b3FiZ3A1NTFnMzlueXk0bXd2cWx1OG4zc2NiaCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/GsTGV7iAI9eZa/giphy.gif", width=150)  
                    st.write("Female")
                else:
                    st.image("https://cdn-icons-png.flaticon.com/512/44/44947.png", width=150)  
                    st.write("Unknown")

            with emo_col:
                # Detailed results
                st.markdown("### 🔍 Detailed Emotion Scores")
                for i, emotion in enumerate(EMOTIONS):
                    color = COLOR_DICT.get(emotion, "grey")
                    percent = pred[i] * 100
                    
                    st.markdown(f"""
                    <div class="progress-container">
                        <div class="progress-label" style="color: {color};">{emotion.capitalize()}</div>
                        <div style="flex-grow: 1;">
                            <div style="height: 20px; background: #e9ecef; border-radius: 10px; overflow: hidden;">
                                <div style="height: 100%; width: {percent}%; background: {color}; 
                                    border-radius: 10px; transition: width 1s ease;"></div>
                            </div>
                        </div>
                        <div class="progress-percent" style="color: {color};">{percent:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)

# -------------------- Main App Logic --------------------
st.sidebar.header("Choose Input Method")
input_mode = st.sidebar.radio("Select input:", ["Upload Audio", "Record Audio"])

if input_mode == "Upload Audio":
    st.markdown("""
    <div class="custom-card">
        <h3 style="color: #6C63FF;">Upload Audio File</h3>
        <p>Upload a WAV or MP3 file to analyze the emotional content.</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("📤 Choose an audio file (WAV/MP3)", type=["wav", "mp3"])
    
    if uploaded_file is not None:
        with st.spinner("Processing audio..."):
            temp_path = os.path.join("temp_audio", uploaded_file.name)
            os.makedirs("temp_audio", exist_ok=True)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            st.audio(uploaded_file, format='audio/wav')
            pred, data3, sr = analyze_emotions(temp_path)
            display_results(temp_path, pred, data3, sr)

elif input_mode == "Record Audio":
    st.markdown("""
    <div class="custom-card">
        <h3 style="color: #6C63FF;">Record Live Audio</h3>
        <p>Click the button below to record 5 seconds of audio directly from your microphone.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Animation for recording
    lottie_recording = load_lottieurl("https://assets1.lottiefiles.com/packages/lf20_sk5h1kfn.json")
    st_lottie(lottie_recording, height=150, key="recording_anim")
    
    st.write("🎙️ Click the button to record 5 seconds of audio.")
    
    # Traffic light
    traffic_light_container = st.empty()
    traffic_light_container.markdown(traffic_light("red"), unsafe_allow_html=True)
    
    if st.button("🔴 Start Recording", key="record_button"):
        traffic_light_container.markdown(traffic_light("yellow"), unsafe_allow_html=True)
        
        fs = 44100
        duration = 5
        
        with st.spinner(f"Recording for {duration} seconds..."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
            
            for i in range(duration):
                time.sleep(1)
                progress_bar.progress((i + 1) / duration)
                status_text.text(f"Recording... {i + 1}/{duration} seconds")
            
            sd.wait()
        
        traffic_light_container.markdown(traffic_light("green"), unsafe_allow_html=True)
        
        with st.spinner("Saving recording..."):
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            sf.write(temp_file.name, audio, fs)
            
            st.success("Recording complete!")
            st.audio(temp_file.name)
            
            pred, data3, sr = analyze_emotions(temp_file.name)
            display_results(temp_file.name, pred, data3, sr)
