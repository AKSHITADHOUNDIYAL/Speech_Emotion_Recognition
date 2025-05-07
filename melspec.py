import numpy as np
import cv2
import librosa
import librosa.display
from tensorflow.keras.models import load_model
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Constants
starttime = datetime.now()

CAT6 = ['fear', 'angry', 'neutral', 'happy', 'sad', 'surprise']
CAT7 = ['fear', 'disgust', 'neutral', 'happy', 'sad', 'surprise', 'angry']
CAT3 = ["positive", "neutral", "negative"]

COLOR_DICT = {
    "neutral": "grey", "positive": "green", "happy": "green",
    "surprise": "orange", "fear": "purple", "negative": "red",
    "angry": "red", "sad": "lightblue", "disgust": "brown"
}

EMOJI_DICT = {
    "happy": "😊", "sad": "😢", "angry": "😠",
    "neutral": "😐", "fear": "😨", "surprise": "😲",
    "disgust": "🤢", "positive": "😁", "negative": "😞"
}

def get_melspec(audio):
    y, sr = librosa.load(audio, sr=44100)
    X = librosa.stft(y)
    Xdb = librosa.amplitude_to_db(abs(X))
    img = np.stack((Xdb,) * 3, -1)
    img = img.astype(np.uint8)
    grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grayImage = cv2.resize(grayImage, (224, 224))
    rgbImage = np.repeat(grayImage[..., np.newaxis], 3, -1)
    return (rgbImage, Xdb, y, sr)

def get_title(predictions, categories, first_line=''):
    emotion = categories[predictions.argmax()]
    confidence = predictions.max() * 100
    emoji = EMOJI_DICT.get(emotion, "")
    txt = f"{first_line}Detected Emotion: {emotion.upper()} {emoji} - {confidence:.2f}%"
    return txt

def draw_audio_status_light(ax, status_color):
    ax.axis('off')
    light = patches.Circle((0.5, 0.5), 0.3, facecolor=status_color, edgecolor='black')
    ax.add_patch(light)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Audio Status", fontsize=10)

def plot_colored_polar(fig, predictions, categories, title="", colors=COLOR_DICT):
    N = len(predictions)
    ind = predictions.argmax()
    COLOR = colors.get(categories[ind], "grey")
    sector_colors = [colors.get(cat, "grey") for cat in categories]

    ax = fig.add_subplot(2, 3, 1, polar=True)
    theta = np.linspace(0.0, 2 * np.pi, N, endpoint=False)

    width = (2 * np.pi) / N * 0.8  # slightly smaller than full width for spacing
    for i in range(N):
        ax.bar(theta[i], predictions[i], width=width, bottom=0.0,
               color=sector_colors[i], alpha=0.25, edgecolor='black')

    # Smooth polar line
    angles = np.concatenate((theta, [theta[0]]))
    data = np.concatenate((predictions, [predictions[0]]))
    ax.plot(angles, data, color=COLOR, linewidth=2)
    ax.fill(angles, data, facecolor=COLOR, alpha=0.25)

    # Set ticks and labels
    ax.set_xticks(theta)
    
    # Ensure tick_labels matches the length of the categories
    tick_labels = [f"{cat}\n{EMOJI_DICT.get(cat, '')}" for cat in categories[:N]]
    ax.set_xticklabels(tick_labels, fontsize=8)

    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['0.25', '0.5', '0.75', '1.0'], fontsize=7)

    ax.set_title(title, fontsize=9, color=COLOR)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.spines['polar'].set_color('lightgrey')

def plot_melspec(path, tmodel=None, three=False, CAT3=CAT3, CAT6=CAT6):
    if tmodel is None:
        tmodel = load_model("tmodel_all.h5")

    mel, _, y, sr = get_melspec(path)
    mel = mel.reshape(1, *mel.shape)
    tpred = tmodel.predict(mel)[0]
    cat = CAT6

    if three:
        pos = tpred[3] + tpred[5] * 0.5
        neu = tpred[2] + tpred[5] * 0.5 + tpred[4] * 0.5
        neg = tpred[0] + tpred[1] + tpred[4] * 0.5
        tpred = np.array([pos, neu, neg])
        cat = CAT3

    # Traffic light status
    max_val = tpred.max()
    light_color = 'green' if max_val > 0.7 else 'yellow' if max_val > 0.4 else 'red'
    title_txt = get_title(tpred, cat)

    fig = plt.figure(figsize=(16, 9))

    # Plot polar chart
    plot_colored_polar(fig, tpred, cat)

    # Bar chart - positioned alongside polar chart
    ax2 = fig.add_subplot(2, 3, 2)
    bar_colors = [COLOR_DICT.get(c, "grey") for c in cat]
    bar_labels = [f"{c} {EMOJI_DICT.get(c, '')}" for c in cat]

    ax2.bar(bar_labels, tpred, color=bar_colors)
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("Probability")
    ax2.set_title("Emotion Bar Chart")
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.set_xticklabels(bar_labels, rotation=45, ha='right')

    # Audio status below polar chart
    ax3 = fig.add_subplot(2, 3, 4)
    draw_audio_status_light(ax3, status_color=light_color)

    # Plot Waveform bottom center
    ax4 = fig.add_subplot(2, 3, 5)
    librosa.display.waveshow(y, sr=sr, ax=ax4)
    ax4.set_title("Waveform", fontsize=10)

    # Plot MFCC bottom right
    ax5 = fig.add_subplot(2, 3, 6)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    librosa.display.specshow(mfcc, x_axis='time', sr=sr, ax=ax5)
    ax5.set_title("MFCC", fontsize=10)

    fig.suptitle("Speech Emotion Recognition\n" + title_txt, fontsize=14, color="darkblue")
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    return fig, tpred


# Run test
if __name__ == "__main__":
    fig, _ = plot_melspec("test.wav")
    plt.show()
