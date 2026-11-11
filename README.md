
# Speech Emotion Recognition and Analysis with an AI Agent

## Project Overview

Speech Emotion Recognition and Analysis with an AI Agent is a machine learning-based application that identifies human emotions from speech audio.

The system uses audio signal processing and a Support Vector Machine (SVM) model to recognize emotions from voice recordings. An AI Agent provides a suitable response based on the detected emotion.

## Objectives

- To identify emotions from speech audio.
- To extract important audio features using MFCC and pitch.
- To train a machine learning model for emotion classification.
- To develop an interactive application using Streamlit.
- To provide emotion-based responses through an AI Agent.

## Technologies Used

- Python
- Google Colab
- Librosa
- NumPy
- Pandas
- Scikit-learn
- Streamlit
- Matplotlib
- Joblib

## Dataset

The project uses the RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song) dataset.

The model recognizes the following eight emotions:

1. Neutral
2. Calm
3. Happy
4. Sad
5. Angry
6. Fearful
7. Disgust
8. Surprised

## Machine Learning Algorithm

Support Vector Machine (SVM) with an RBF kernel is used for emotion classification.

### Feature Extraction

The following audio features are extracted:

- MFCC (Mel Frequency Cepstral Coefficients)
- Pitch mean
- Pitch standard deviation
- Energy

The extracted features are standardized using StandardScaler.

## Project Structure

```text
Speech_Emotion_Project/
│
├── app.py
├── emotion_model.pkl
├── label_encoder.pkl
├── requirements.txt
└── README.md
```

## Installation

Install the required Python libraries:

```bash
pip install -r requirements.txt
```

## Running the Application

Run the following command:

```bash
streamlit run app.py
```

The application will open in your web browser.

## Application Features

- Upload speech audio in WAV format.
- Analyze the uploaded audio.
- Predict the speaker's emotion.
- Display prediction confidence.
- Visualize emotion probabilities.
- Generate an AI Agent response based on the predicted emotion.

## Model Training

The machine learning model is trained using audio features extracted from the dataset.

The trained model and label encoder are saved as:

- emotion_model.pkl
- label_encoder.pkl

These files are loaded by the Streamlit application for emotion prediction.

## Expected Output

The application displays:

- Predicted emotion
- Prediction confidence
- Emotion probability graph
- AI Agent response

## Future Enhancements

- Support for additional audio formats.
- Real-time speech emotion recognition.
- Improved model accuracy.
- Multilingual speech emotion recognition.
- Advanced conversational AI Agent integration.

## Disclaimer

This application identifies acoustic patterns associated with emotions in speech. It does not reliably determine a person's actual emotional or mental state and is not a medical or psychological diagnostic tool.

## Conclusion

This project demonstrates how machine learning and audio processing can be combined to recognize speech emotions and provide interactive responses through an AI Agent.
