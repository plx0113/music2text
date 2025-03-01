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
    target_dir = "music_genres_classification"
    os.makedirs(target_dir, exist_ok=True)
    download_file(MODEL_URL, os.path.join(target_dir, "model.safetensors"))
    download_file(PYTORCH_URL, os.path.join(target_dir, "pytorch_model.bin"))

def safe_float(value):
    """Safely convert numpy values to float."""
    try:
        if isinstance(value, (np.ndarray, np.generic)):
            return float(value.item())
        return float(value)
    except (ValueError, TypeError, AttributeError):
        return 0.0  # fallback value

# ==========================================================
# Custom Micro-Genre Generation Functions
# ==========================================================
custom_genre_rules = [
    {"conditions": {"pop": 0.7, "electronic": 0.1}, "genre": "Electropop"},
    {"conditions": {"pop": 0.7, "hiphop": 0.1}, "genre": "PopRap"},
    {"conditions": {"rock": 0.6, "metal": 0.3}, "genre": "Heavy Rock"},
    {"conditions": {"jazz": 0.5, "blues": 0.3}, "genre": "JazzBlues"},
    {"conditions": {"rnb": 0.4, "hiphop": 0.3}, "genre": "RnbHop"}
]

def compute_normalized_top_genres(probabilities, top_n=4):

    # Otherwise, process the top N genres as before
    sorted_genres = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
    top_genres = dict(sorted_genres[:top_n])
    total_score = sum(top_genres.values())
    if total_score > 0:
        normalized = {genre: score / total_score for genre, score in top_genres.items()}
    else:
        normalized = top_genres
    return normalized

def determine_micro_genre(probabilities):
    for rule in custom_genre_rules:
        if all(probabilities.get(genre, 0) >= threshold for genre, threshold in rule["conditions"].items()):
            return rule["genre"]
    return "Undefined"
# ==========================================================
# End Custom Micro-Genre Generation Functions
# ==========================================================

# --- Load Funky Genres from JSON file ---
try:
    with open("funky_genres.json", "r", encoding="utf-8", errors="replace") as f:
        funky_genres = json.load(f)
    # Format the structured data into a readable string for the prompt
    funky_genres_str = ""
    for parent, subgenres in funky_genres.items():
        funky_genres_str += f"{parent}:\n" + "\n".join(subgenres) + "\n\n"
except Exception as e:
    logging.error("Error loading funky_genres.json: " + str(e))
    funky_genres_str = "Astro Acid Breaks\nCosmic 8-Bit Garage\nNebula Nu Metal"  # fallback

# Truncate the audio before conversion
def convert_to_wav(audio_file, temp_dir, sr=44100, max_duration=30):
    try:
        file_extension = audio_file.name.split(".")[-1].lower()
        input_file_path = os.path.join(temp_dir, "input." + file_extension)
        with open(input_file_path, "wb") as f:
            f.write(audio_file.read())
        audio = AudioSegment.from_file(input_file_path, format=file_extension)
        
        # Clip the segment from 30 seconds to 60 seconds (pydub uses millisecond indexing)
        start_ms = 30 * 1000
        end_ms = start_ms + (max_duration * 1000)
        truncated_audio = audio[start_ms:end_ms]
        
        wav_file_path = os.path.join(temp_dir, "output.wav")
        truncated_audio.export(wav_file_path, format="wav", parameters=["-ar", str(sr)])
        return wav_file_path
    except Exception as e:
        logging.error(f"Conversion to WAV error: {e}")
        return None


def extract_audio_features(audio_file_path):
    try:
        y, sr = librosa.load(audio_file_path, sr=None, mono=True, dtype='float32')
        logging.info(f"Librosa loaded audio: {audio_file_path} with sample rate {sr} and shape {y.shape}")
        
        if y is None or len(y) == 0:
            raise ValueError("Librosa failed: Empty or unreadable file")

        # BPM Detection with error handling
        try:
            onset_env = librosa.onset.onset_strength(y=y, sr=sr, aggregate=np.median)
            tempo, beats = librosa.beat.beat_track(
                onset_envelope=onset_env, 
                sr=sr,
                hop_length=512,
                start_bpm=120,
                tightness=100
            )
            
            if len(beats) > 0:
                beat_strength = librosa.util.normalize(onset_env)
                confidence = float(np.mean(beat_strength[beats]))  # Convert to float
                logging.info(f"Beat detection confidence: {confidence:.2f}")
                
                if confidence < 0.4:
                    alt_tempo, alt_beats = librosa.beat.beat_track(
                        onset_envelope=onset_env, 
                        sr=sr,
                        hop_length=512,
                        start_bpm=float(tempo)*0.5  # Convert tempo to float
                    )
                    if len(alt_beats) > 0:
                        alt_confidence = float(np.mean(librosa.util.normalize(onset_env)[alt_beats]))
                        if alt_confidence > confidence * 1.2:
                            tempo = float(alt_tempo)  # Convert to float
                            beats = alt_beats
                            confidence = alt_confidence
                            logging.info(f"Using half-tempo: {tempo:.1f} BPM with confidence {confidence:.2f}")
            
            tempo = float(tempo)  # Ensure tempo is a float
            # Adjust BPM if it is out of the desired range
            if tempo < 70:
                tempo = tempo * 2
                logging.info(f"Detected BPM below 70, doubling BPM to: {tempo:.1f}")
            elif tempo > 180:
                tempo = tempo / 2
                logging.info(f"Detected BPM above 180, halving BPM to: {tempo:.1f}")
    
        except Exception as e:
            logging.error(f"Error in tempo detection: {str(e)}")
            tempo = 120.0  # fallback tempo

        # Key Detection
        try:
            chromatic_scale = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=12)
            y_harmonic, _ = librosa.effects.hpss(y)
            
            # Get initial key
            tonal = Tonal_Fragment(y, sr)
            key = tonal.key
            
            # Additional features with error handling
            rms = librosa.feature.rms(y=y)[0]
            articulation_rate = float(np.mean(onset_env))
            spectral_centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
            spectral_bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))
            
            # HPSS decomposition
            y_harmonic, y_percussive = librosa.effects.hpss(y)
            
            # MFCCs
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=33)
            
            # Convert numpy values to Python native types
            return (
                float(tempo),
                str(key),
                rms.astype(float),  # Convert to float array
                float(articulation_rate),
                float(spectral_centroid),
                float(spectral_bandwidth),
                y,
                int(sr),
                y_harmonic,
                y_percussive,
                mfccs
            )
            
        except Exception as e:
            logging.error(f"Error in feature extraction: {str(e)}")
            raise
            
    except Exception as e:
        logging.error(f"Feature extraction error: {str(e)}")
        return (None,) * 11

def convert_numpy_data(obj):
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
    segment_length = segment_duration * sr
    num_segments = len(y) // segment_length
    segments = [y[i * segment_length:(i + 1) * segment_length] for i in range(num_segments)]
    return segments

def call_openai_with_retry(messages, model="gpt-4", max_retries=5, initial_delay=1):
    for attempt in range(max_retries):
        try:
            response = openai.ChatCompletion.create(model=model, messages=messages)
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
    loop = asyncio.get_running_loop()
    response = await loop.run_in_executor(None, call_openai_with_retry, messages, model)
    return response

@st.cache_resource
def load_audio_classifier():
    from transformers import pipeline
    try:
        classifier = pipeline("audio-classification", model="dima806/music_genres_classification")
        logging.info("Hugging Face pipeline classifier loaded successfully.")
        return classifier
    except Exception as e:
        logging.error(f"Error loading Hugging Face pipeline classifier: {e}")
        return None

audio_classifier_hf = load_audio_classifier()

try:
    model_path = "/root/music2text/music_genres_classification"
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

    # Initialize progress bar (0 to 100)
    progress_bar = st.progress(0)
    
    audio_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "flac", "m4a", "ogg"])
    
    if audio_file is not None:
    # Extract just the song name without extension
        track_title = os.path.splitext(audio_file.name)[0]
        with st.spinner("Processing audio..."):
            status_text = st.empty()
            try:
                with tempfile.TemporaryDirectory() as temp_dir:
                    logging.info(f"Created temporary directory: {temp_dir}")
                    status_text.text("Listening...")
    
                    # --- Step 1: Convert to WAV ---
                    wav_file_path = convert_to_wav(audio_file, temp_dir)
                    if not wav_file_path:
                        st.error("Failed to convert audio to WAV.")
                        raise Exception("Failed to convert audio to WAV.")
                    progress_bar.progress(20)
    
                    # --- Step 2: Extract Audio Features ---
                    with ThreadPoolExecutor(max_workers=1) as executor:
                        future = executor.submit(extract_audio_features, wav_file_path)
                        (tempo, key, rms, articulation_rate, spectral_centroid, spectral_bandwidth, y, sr, y_harmonic, y_percussive, mfccs) = future.result()
    
                    audio_features_extracted = tempo is not None and key is not None
                    if audio_features_extracted:
                        st.write(f"Estimated Tempo: {tempo:.0f} BPM")
                        st.write(f"Detected Key: {key}")
                    else:
                        st.warning("Could not extract audio features.")
                        st.stop()
                    progress_bar.progress(40)
    
                    # --- Step 3: Genre Classification ---
                    if audio_classifier_hf is not None:
                        hf_predictions = audio_classifier_hf(wav_file_path)
                        genre_mapping = {"disco": "electronic"}
                        genre_scores = {}
                        for pred in hf_predictions:
                            label = pred['label'].lower()
                            label = genre_mapping.get(label, label)
                            genre_scores[label] = pred['score']
                        normalized_genres = compute_normalized_top_genres(genre_scores, top_n=4)
                        normalized_genres = convert_numpy_data(normalized_genres)  # Ensure pure Python types
                        progress_bar.progress(60)

                        # Extract track title without file extension and in lowercase for comparisons
                        track_title = os.path.splitext(audio_file.name)[0]
                        track_title_lower = track_title.lower()

                        # Use the track title to decide whether to remap reggae to rnb.
                        if "reggae" in normalized_genres:
                            if ("reggae" not in track_title_lower and "dub" not in track_title_lower) and normalized_genres["reggae"] < 0.925:
                            # Remap reggae score to rnb
                                normalized_genres["rnb"] = normalized_genres.pop("reggae")
                        
                        # Print macro genre scores for debugging
                        st.write("Normalized Macro Genre Scores:", normalized_genres)        

                        # --- Feedback Loop: Generate Final Micro-Genre via ChatGPT ---
                        prompt_for_micro_genre = f"""
You are a music genre expert with deep musical knowledge across mainstream and micro-genres. Given these inputs:

    Normalized Genre Scores (JSON): {normalized_genres}
    BPM: {tempo} (BPM values may be doubled or halved as needed)
    Track Title: "{track_title}" (use any recognizable cues from the title, if you recognize the song use it's genre)
    Funky Genre Database: {funky_genres_str}

Please generate the final micro-genre for this track. Feel free to get whacky and creative combining words and genres but keep it rooted in the macro-genres and Funky Genre Database. Output only the final micro-genre as a concise string.
"""
                        response_genre = call_openai_with_retry([{"role": "system", "content": prompt_for_micro_genre}])
                        final_micro_genre = response_genre["choices"][0]["message"]["content"].strip()
                        progress_bar.progress(80)
                        # --- End Feedback Loop ---
    
                        # --- Step 4: Prepare and Display Analysis ---
                        track_title = os.path.splitext(audio_file.name)[0]
                        dynamics_range = float(np.max(rms) - np.min(rms)) if rms is not None else None
    
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
    
                        feature_summary = {
                            "tempo": tempo,
                            "key": key,
                            "Final Micro-Genre": final_micro_genre,
                            "track_title": track_title,
                            "articulation_rate": articulation_rate,
                            "dynamics_range": dynamics_range,
                            "spectral_centroid": spectral_centroid,
                            "spectral_bandwidth": spectral_bandwidth,
                        }
                        audio_analysis = {"features": convert_numpy_data(feature_summary)}
    
                        prompt_for_analysis = f"""
Data:
- Final Micro-Genre: {final_micro_genre}
- File Name: {track_title}
- Articulation Rate: {articulation_rate:.2f}
- Dynamics Range: {dynamics_range:.2f}
- Spectral Centroid: {spectral_centroid:.2f}
- Spectral Bandwidth: {spectral_bandwidth:.2f}

Instructions:
1. Begin by explicitly stating the genre "{final_micro_genre}".
2. Pretend you are a music critic for Pitchfork magazine, but never mention Pitchfork.
3. If you recognize the song, talk about it's lyrical content, if you don't recognize the song, talk about the instrumentation.
4. Write a single, concise, and evocative paragraph reviewing the track—focusing on its style, genre influences, and emotional impact.
5. Do not mention technical details such as BPM or key.
"""

                        response_analysis = call_openai_with_retry([
                            {"role": "system", "content": prompt_for_analysis},
                            {"role": "user", "content": json.dumps(audio_analysis)}
                        ])
                        st.write("AI Analysis:", response_analysis["choices"][0]["message"]["content"])
                        progress_bar.progress(100)
                    else:
                        st.warning("Genre classifier could not be loaded.")
            except Exception as e:
                st.error(f"An error occurred: {e}")
            finally:
                status_text.empty()
                logging.info("Processing complete (temporary directory cleaned up automatically).")
