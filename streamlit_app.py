import streamlit as st
import numpy as np
import librosa
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder

# Load the trained model and label encoder
model = load_model('trained_model.h5')
labelencoder = LabelEncoder()
labelencoder.classes_ = np.load('classes.npy', allow_pickle=True)

# Function to extract features from the audio file
def extract_features(audio_file):
    y, sr = librosa.load(audio_file, res_type='kaiser_fast')
    mfccs_features = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
    return mfccs_scaled_features

# Main function
def main(title="Music Genre Classification".upper()):
    st.title(title)
    st.write("This application classifies music genres using a Convolutional Neural Network (CNN).")
    
    # File upload
    uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])
    
    if uploaded_file is not None:
        st.audio(uploaded_file, format='audio/wav')
        
        with st.spinner('Extracting features...'):
            features = extract_features(uploaded_file)
            features = features.reshape(1, -1)  # Reshape to match model input
        
        with st.spinner('Classifying genre...'):
            predicted_probabilities = model.predict(features)
            predicted_label = np.argmax(predicted_probabilities, axis=1)
            predicted_class = labelencoder.inverse_transform(predicted_label)
        
        st.success(f"The predicted genre is: **{predicted_class[0]}**")

# Run the app
if __name__ == '__main__':
    main()
