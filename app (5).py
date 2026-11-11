import streamlit as st
import librosa
import numpy as np
import joblib
import tempfile
import os
import matplotlib.pyplot as plt

# Load trained model and encoder
model = joblib.load("emotion_model.pkl")
encoder = joblib.load("label_encoder.pkl")

st.set_page_config(
    page_title="Speech Emotion Recognition",
    page_icon="🎙️",
    layout="centered"
)

st.title("🎙️ Speech Emotion Recognition")
st.subheader("AI-Based Speech Emotion Analysis")

st.write(
    "Upload a voice recording to identify the emotion "
    "expressed through its acoustic features."
)

# Extract the same features used during training
def extract_features(file_path):
    audio, sr = librosa.load(
        file_path, sr=22050, duration=3
    )

    mfcc = librosa.feature.mfcc(
        y=audio, sr=sr, n_mfcc=40
    )

    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    pitch = librosa.yin(
        audio, fmin=50, fmax=500, sr=sr
    )

    pitch_mean = np.mean(pitch)
    pitch_std = np.std(pitch)

    energy = librosa.feature.rms(y=audio)

    features = np.concatenate([
        mfcc_mean,
        mfcc_std,
        [pitch_mean, pitch_std, np.mean(energy)]
    ])

    return features

# AI agent response
def emotion_agent(emotion):
    responses = {
        "happy": "You sound cheerful! Keep enjoying the positive moments.",
        "sad": "Your voice contains cues of sadness. Take a moment to relax.",
        "angry": "Your voice contains cues of anger. Try taking a few slow breaths.",
        "fearful": "Your voice contains cues of fear. Take a moment to relax.",
        "disgust": "Your voice contains cues associated with disgust.",
        "surprised": "Your voice contains cues associated with surprise.",
        "neutral": "Your voice sounds relatively neutral.",
        "calm": "Your voice sounds calm and relaxed."
    }

    return responses.get(
        emotion,
        "Unable to identify the emotion."
    )

uploaded_file = st.file_uploader(
    "Upload an audio file",
    type=["wav"]
)

if uploaded_file is not None:

    st.audio(uploaded_file)

    if st.button("Analyze Emotion"):

        temp_path = None

        try:
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=".wav"
            ) as temp_file:

                temp_file.write(uploaded_file.getvalue())
                temp_path = temp_file.name

            features = extract_features(temp_path)

            prediction = model.predict([features])[0]
            emotion = encoder.inverse_transform(
                [prediction]
            )[0]

            probabilities = model.predict_proba(
                [features]
            )[0]

            confidence = np.max(probabilities) * 100

            st.success("Analysis Completed!")

            st.subheader("Predicted Emotion")
            st.title(f"😊 {emotion.capitalize()}")

            st.write(
                f"Estimated model confidence: {confidence:.2f}%"
            )

            st.subheader("Emotion Probability Analysis")

            fig, ax = plt.subplots(figsize=(9, 4))
            ax.bar(
                encoder.classes_,
                probabilities * 100
            )
            ax.set_xlabel("Emotions")
            ax.set_ylabel("Probability (%)")
            ax.tick_params(axis="x", rotation=35)

            st.pyplot(fig)

            st.subheader("🤖 AI Agent Response")
            st.info(emotion_agent(emotion))

            st.caption(
                "This system identifies acoustic patterns in speech. "
                "It does not reliably determine a person's actual "
                "emotional or psychological state."
            )

        except Exception as e:
            st.error(f"Error during analysis: {e}")

        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
