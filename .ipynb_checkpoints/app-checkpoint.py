import streamlit as st
import joblib
import librosa
import numpy as np

# Load the model
spam_clf = joblib.load(open('trained_model.h5', 'rb'))

# Function to extract features from the audio file
def extract_features(audio_file):
    y, sr = librosa.load(audio_file, duration=30)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
    spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr).T, axis=0)
    spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr).T, axis=0)
    zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y=y).T, axis=0)
    features = np.hstack([mfcc, chroma, spectral_contrast, spectral_rolloff, zero_crossing_rate])
    return features

# Main function
def main(title="Music Genre Classification".upper()):
    st.title(title)
    st.write("This application classifies music genres using a Convolutional Neural Network (CNN).")
    
    # File upload
    uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "ogg"])
    
    if uploaded_file is not None:
        st.audio(uploaded_file, format='audio/wav')
        
        with st.spinner('Extracting features...'):
            features = extract_features(uploaded_file)
            features = features.reshape(1, -1)
        
        with st.spinner('Classifying genre...'):
            prediction = spam_clf.predict(features)
            genre = prediction[0]
        
        st.success(f"The predicted genre is: **{genre}**")

# Run the app
if __name__ == '__main__':
    main()
