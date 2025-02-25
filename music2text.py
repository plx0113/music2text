import streamlit as st
import librosa
import librosa.display
import numpy as np
import openai
import json
from keyfinder import Tonal_Fragment
from pydub import AudioSegment
import os
import logging
import tempfile
import time
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import pickle
from sklearn.preprocessing import StandardScaler
from transformers import pipeline  # For the HF pipeline
import torch
import math
from transformers import Wav2Vec2FeatureExtractor, AutoModelForAudioClassification
import requests
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set your OpenAI API key from the environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("No OPENAI_API_KEY environment variable found.")

# DigitalOcean Spaces URLs for your models
MODEL_URL = "https://music2text-models.nyc3.digitaloceanspaces.com/genres_classification/model.safetensors"
PYTORCH_URL = "https://music2text-models.nyc3.digitaloceanspaces.com/genres_classification/pytorch_model.bin"

def download_file(url, local_filename):
    if os.path.exists(local_filename):
        print(f"{local_filename} already exists, skipping download.")
        return
    print(f"Downloading {local_filename}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(local_filename, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print("Download complete.")

def load_models():
    # Ensure the target directory exists
    target_dir = "music_genres_classification"
    os.makedirs(target_dir, exist_ok=True)

    # Download files into the target directory from DigitalOcean Spaces
    download_file(MODEL_URL, os.path.join(target_dir, "model.safetensors"))
    download_file(PYTORCH_URL, os.path.join(target_dir, "pytorch_model.bin"))

def safe_float(value):
    """
    Safely convert a scalar or array-like numeric to float,
    avoiding NumPy's deprecation warning.
    """
    if isinstance(value, (np.ndarray, np.generic)) and value.size == 1:
        return value.item()
    return float(value)

# Truncate the audio before conversion
def convert_to_wav(audio_file, temp_dir, sr=44100, max_duration=30):
    """Converts the uploaded audio file to WAV format using pydub and truncates to max_duration."""
    try:
        file_extension = audio_file.name.split(".")[-1].lower()
        input_file_path = os.path.join(temp_dir, "input." + file_extension)
        with open(input_file_path, "wb") as f:
            f.write(audio_file.read())
        audio = AudioSegment.from_file(input_file_path, format=file_extension)

        # Truncate audio to 30 seconds before converting
        truncated_audio = audio[:max_duration * 1000]  # pydub works in milliseconds
        wav_file_path = os.path.join(temp_dir, "output.wav")
        truncated_audio.export(wav_file_path, format="wav", parameters=["-ar", str(sr)])
        return wav_file_path
    except Exception as e:
        logging.error(f"Conversion to WAV error: {e}")
        return None

def extract_audio_features(audio_file_path):
    """Extracts key, tempo, and root chroma from the audio file along with other features."""
    try:
        # Updated line: mono=True, dtype='float32'
        y, sr = librosa.load(audio_file_path, sr=None, mono=True, dtype='float32')
        logging.info(f"Librosa loaded audio: {audio_file_path} with sample rate {sr} and shape {y.shape}")

        if y is None or len(y) == 0:
            raise ValueError("Librosa failed: Empty or unreadable file")

        # IMPROVED BPM DETECTION
        # 1. Calculate onset envelope with different settings
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, aggregate=np.median)
        
        # 2. Detect tempo with better parameters
        tempo, beats = librosa.beat.beat_track(
            onset_envelope=onset_env, 
            sr=sr,
            hop_length=512,
            start_bpm=120,
            tightness=100
        )
        
        # 3. Calculate confidence
        if len(beats) > 0:
            beat_strength = librosa.util.normalize(onset_env)
            confidence = np.mean(beat_strength[beats])
            logging.info(f"Beat detection confidence: {confidence:.2f}")
            
            # 4. Handle tempo octave errors
            # If confidence is low, check alternative tempos
            if confidence < 0.4:
                alt_tempo, alt_beats = librosa.beat.beat_track(
                    onset_envelope=onset_env, 
                    sr=sr,
                    hop_length=512,
                    start_bpm=tempo*0.5  # Try half tempo
                )
                alt_confidence = np.mean(librosa.util.normalize(onset_env)[alt_beats]) if len(alt_beats) > 0 else 0
                
                if alt_confidence > confidence * 1.2:  # If significantly better
                    tempo, beats, confidence = alt_tempo, alt_beats, alt_confidence
                    logging.info(f"Using half-tempo: {tempo:.1f} BPM with confidence {confidence:.2f}")
        
        tempo = safe_float(tempo)

        # IMPROVED KEY DETECTION
        
        chromatic_scale = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

        # 1. Use multiple chroma variants for better pitch representation
        chroma_cqt = librosa.feature.chroma_cqt(y=y, sr=sr, bins_per_octave=36)
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=12)
        
        # 2. Create more accurate harmonic-based chroma
        y_harmonic, _ = librosa.effects.hpss(y)
        chroma_harm = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr)
        
        # 3. Consensus-based key detection
        keys = []
        confidences = []
        
        # Original key detection
        tonal = Tonal_Fragment(y, sr)
        key1 = tonal.key
        keys.append(key1)
        
        # Add Krumhansl-Schmuckler key finding algorithm
        chroma_avg = np.mean(chroma_stft, axis=1)
        major_profiles = np.array([
            [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],  # C Major
        ])
        minor_profiles = np.array([
            [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17],  # A Minor
        ])
        
        # Rotate profiles to create all key templates
        all_keys = []
        for i in range(12):
            all_keys.extend([
                np.roll(major_profiles[0], i),
                np.roll(minor_profiles[0], i)
            ])
        
        key_names = []
        for i in range(12):
            key_names.append(f"{chromatic_scale[i]} Major")
            key_names.append(f"{chromatic_scale[(i+9)%12]} Minor")
        
        # Calculate correlation with each key profile
        correlations = []
        for profile in all_keys:
            corr = np.corrcoef(chroma_avg, profile)[0, 1]
            correlations.append(corr)
        
        best_key_idx = np.argmax(correlations)
        best_correlation = correlations[best_key_idx]
        key2 = key_names[best_key_idx]
        keys.append(key2)
        confidences.append(best_correlation)
        
        # 4. Add segment-based key analysis
        if len(y) > sr * 5:  # If more than 5 seconds
            segment_length = sr * 5
            num_segments = len(y) // segment_length
            segment_keys = []
            
            # Analyze non-overlapping 5-second segments
            for i in range(min(num_segments, 3)):  # Analyze up to 3 segments
                segment = y[i * segment_length:(i + 1) * segment_length]
                segment_chroma = librosa.feature.chroma_cqt(y=segment, sr=sr)
                segment_chroma_avg = np.mean(segment_chroma, axis=1)
                
                # Calculate correlation with each key profile
                segment_correlations = []
                for profile in all_keys:
                    corr = np.corrcoef(segment_chroma_avg, profile)[0, 1]
                    segment_correlations.append(corr)
                
                best_segment_key_idx = np.argmax(segment_correlations)
                segment_keys.append(key_names[best_segment_key_idx])
            
            # Add most common segment key
            if segment_keys:
                from collections import Counter
                most_common_key = Counter(segment_keys).most_common(1)[0][0]
                keys.append(most_common_key)
        
        # Final key determination by voting
        if len(keys) > 0:
            from collections import Counter
            key_counter = Counter(keys)
            key = key_counter.most_common(1)[0][0]
            logging.info(f"Key detection candidates: {keys}")
            logging.info(f"Final key selected: {key}")
        else:
            key = key1  # Fallback to original

        # Rest of the function remains the same...
        rms = librosa.feature.rms(y=y)[0]
        try:
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            articulation_rate = np.mean(onset_env)
        except Exception as e:
            logging.error(f"Error calculating articulation rate: {e}")
            articulation_rate = None

        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=33)
        return (tempo, key, rms, articulation_rate, 
                spectral_centroid, spectral_bandwidth, y, sr, 
                y_harmonic, y_percussive, mfccs)
    except Exception as e:
        logging.error(f"Feature extraction error: {e}")
        return (None,)*11

def convert_numpy_data(obj):
    """Converts numpy data types to serializable Python types."""
    import numpy as np
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, list):
        return [convert_numpy_data(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_numpy_data(value) for key, value in obj.items()}
    else:
        return obj

def segment_audio(y, sr, segment_duration=10):
    """Segments audio into smaller parts for arrangement analysis."""
    segment_length = segment_duration * sr
    num_segments = len(y) // segment_length
    segments = [y[i * segment_length:(i + 1) * segment_length] for i in range(num_segments)]
    return segments

def call_openai_with_retry(messages, model="gpt-4", max_retries=5, initial_delay=1):
    """Calls OpenAI with retry logic and exponential backoff."""
    for attempt in range(max_retries):
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages
            )
            return response
        except Exception as e:
            logging.warning(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                delay = initial_delay * (2 ** attempt)
                logging.info(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                logging.error("Max retries reached. Aborting.")
                raise e

async def async_openai_call(messages, model="gpt-4"):
    """Asynchronous wrapper for OpenAI API call using run_in_executor."""
    loop = asyncio.get_running_loop()
    response = await loop.run_in_executor(None, call_openai_with_retry, messages, model)
    return response

@st.cache_resource
def load_audio_classifier():
    """Loads the Hugging Face pipeline for audio classification from a local model."""
    from transformers import pipeline
    try:
        classifier = pipeline("audio-classification", model="dima806/music_genres_classification")
        logging.info("Hugging Face pipeline classifier loaded successfully.")
        return classifier
    except Exception as e:
        logging.error(f"Error loading Hugging Face pipeline classifier: {e}")
        return None

audio_classifier_hf = load_audio_classifier()

# Load the audio classification model locally from the DigitalOcean downloaded files
try:
    model_path = "/root/music2text/music_genres_classification"  # Ensure this directory exists locally
    processor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)
    model = AutoModelForAudioClassification.from_pretrained(model_path, trust_remote_code=True).to("cpu")
    st.success("Model successfully loaded from local directory.")
except Exception as e:
    st.error(f"Error loading model directly: {e}")
    processor = None
    model = None

if __name__ == "__main__":
    st.title("Music to Text App")
    load_models()

    audio_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "flac", "m4a", "ogg"])

    if audio_file is not None:
        with st.spinner("Processing audio..."):
            status_text = st.empty()
            try:
                with tempfile.TemporaryDirectory() as temp_dir:
                    logging.info(f"Created temporary directory: {temp_dir}")
                    status_text.text("Listening...")

                    # 1. Convert uploaded file to WAV
                    wav_file_path = convert_to_wav(audio_file, temp_dir)
                    if not wav_file_path:
                        st.error("Failed to convert audio to WAV.")
                        raise Exception("Failed to convert audio to WAV.")

                    # 2. Extract audio features using a separate thread (CPU-bound)
                    with ThreadPoolExecutor(max_workers=1) as executor:
                        future = executor.submit(extract_audio_features, wav_file_path)
                        (tempo, key, rms, articulation_rate, 
                         spectral_centroid, spectral_bandwidth, y, sr, 
                         y_harmonic, y_percussive, mfccs) = future.result()

                    audio_features_extracted = tempo is not None and key is not None
                    if audio_features_extracted:
                        st.write(f"Estimated Tempo: {tempo:.0f} BPM")
                        st.write(f"Detected Key: {key}")
                        st.write(f"Articulation Rate: {articulation_rate:.2f}")

                    else:
                        st.warning("Could not extract audio features.")
                        st.stop()

                    # 3. Use Hugging Face pipeline classifier to estimate genre
                    estimated_genre_summary = ""
                    if audio_classifier_hf is not None:
                        hf_predictions = audio_classifier_hf(wav_file_path)
                        genre_mapping = {"disco": "electronic"}
                        estimated_genre_summary = ", ".join(
                            [f"{genre_mapping.get(pred['label'].lower(), pred['label'])} ({pred['score']:.2f})" for pred in hf_predictions]
                        )
                        st.write("Estimated Genre:")
                        st.write(estimated_genre_summary)

                    else:
                        st.warning("Genre classifier could not be loaded.")

                    # 4. Get the file name
                    file_name = audio_file.name

                    # 5. Segment audio for arrangement analysis
                    #if y is not None and sr is not None:
                        #segments = segment_audio(y, sr)
                        #num_segments = len(segments)
                    #else:
                        #segments = []
                        #num_segments = 0
                   # st.write(f"Number of Segments: {num_segments}")

                    # 6. Calculate dynamics range (RMS)
                    if rms is not None:
                        dynamics_range = np.max(rms) - np.min(rms)
                    else:
                        dynamics_range = None

                    # 7. Display spectrogram of the audio
                    if y is not None and sr is not None:
                        fig, ax = plt.subplots()
                        D = np.abs(librosa.stft(y))
                        img = librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max),
                                                       y_axis='log', x_axis='time', sr=sr, ax=ax)
                        ax.set_title('Power Spectrogram')
                        fig.colorbar(img, ax=ax)
                        st.pyplot(fig)
                    else:
                        st.warning("Could not generate spectrogram.")

                    # 8. Prepare feature summary for OpenAI analysis
                    feature_summary = {
                        "tempo": tempo,
                        "key": key,
                        "Estimated Genre": estimated_genre_summary,
                        "file_name": file_name,
                        "articulation_rate": articulation_rate,
                        "dynamics_range": dynamics_range,
                        "spectral_centroid": spectral_centroid,
                        "spectral_bandwidth": spectral_bandwidth,
                    }
                    audio_analysis = {"features": convert_numpy_data(feature_summary)}

                    # 9. Asynchronously call OpenAI to analyze the song
                    async def get_analysis():
                        return await async_openai_call([
                            {
                                "role": "system",
                                "content": f"""
You are a seasoned music analyst with exceptional listening skills. Using the provided data (tempo, key, genre probabilities, articulation rate, and segment info), deduce the song's genre, style, and emotional impact. Adjust for possible tempo doubling/halving, evaluate the song's structure and transitions, and focus on the dominant genre while noting subtle influences. If the title hints at deeper lyrics, discuss themes and key phrases. Provide a comprehensive, coherent interpretation of the song.

- Tempo: {tempo:.0f} BPM
- Key: {key}
- Estimated Genre: {estimated_genre_summary}
- File Name: {file_name}
- Articulation Rate: {articulation_rate:.2f}
- Dynamics Range: {dynamics_range:.2f}
- Spectral Centroid: {spectral_centroid:.2f}
- Spectral Bandwidth: {spectral_bandwidth:.2f}
"""
                            },
                            {
                                "role": "user", "content": json.dumps(audio_analysis)
                            }
                        ])

                    analysis = asyncio.run(get_analysis())
                    st.write("AI Analysis:", analysis["choices"][0]["message"]["content"])
            except Exception as e:
                st.error(f"An error occurred: {e}")
            finally:
                status_text.empty()
                logging.info("Processing complete (temporary directory cleaned up automatically).")
