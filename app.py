import streamlit as st
import numpy as np
import librosa
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder

# Load the trained model and label encoder
model = load_model('trained_model.h5')
labelencoder = LabelEncoder()
labelencoder.classes_ = np.load('classes.npy', allow_pickle=True)

# Main function
def main():
    st.title("Music Genre Classification")

    # File uploader
    uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3"])
    if uploaded_file is not None:
        audio_bytes = uploaded_file.read()
        with open('temp_audio.wav', 'wb') as f:
            f.write(audio_bytes)

        audio, sample_rate = librosa.load('temp_audio.wav', res_type='kaiser_fast')
        mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
        mfccs_scaled_features = mfccs_scaled_features.reshape(1, -1)

        # Predict genre
        predicted_probabilities = model.predict(mfccs_scaled_features)
        predicted_label = np.argmax(predicted_probabilities, axis=1)
        predicted_class = labelencoder.inverse_transform(predicted_label)

        # Display prediction
        st.write(f'Predicted Genre: {predicted_class[0]}')

if __name__ == "__main__":
    main()
