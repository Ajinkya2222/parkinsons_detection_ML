# ==========================================
# PARKINSON'S AUDIO → PREDICTION PIPELINE
# ==========================================

# ---------- Imports ----------
import numpy as np
import pandas as pd
import joblib
import librosa
import opensmile
from pydub import AudioSegment

# ---------- Load trained artifacts ----------
pd_model = joblib.load("pd_xgb_model.pkl")        # or pd_svm_model.pkl
pd_scaler = joblib.load("pd_scaler.pkl")
pd_feature_names = joblib.load("pd_feature_names.pkl")

# ---------- Initialize OpenSMILE ----------
smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals,
)

# ---------- Convert AAC → WAV ----------
def convert_aac_to_wav(audio_path, out_path="pd_converted.wav"):
    audio = AudioSegment.from_file(audio_path)
    audio = audio.set_channels(1).set_frame_rate(16000)
    audio.export(out_path, format="wav")
    return out_path

# ---------- Audio validity check ----------
def is_audio_valid(wav_path, min_duration=1.5):
    y, sr = librosa.load(wav_path, sr=16000)
    duration = len(y) / sr
    energy = np.mean(np.abs(y))
    return duration >= min_duration and energy > 0.001

# ---------- Extract OpenSMILE features safely ----------
def extract_opensmile_features_safe(wav_path):
    if not is_audio_valid(wav_path):
        raise ValueError("Audio too short or silent for Parkinson's analysis")

    df = smile.process_file(wav_path)

    if df is None or df.shape[0] == 0:
        raise ValueError("No voiced segments detected in audio")

    return df

# ---------- Prepare model input ----------
def prepare_pd_input_from_audio(wav_path):
    df = extract_opensmile_features_safe(wav_path)

    X_new = pd.DataFrame(0.0, index=[0], columns=pd_feature_names)

    for col in pd_feature_names:
        if col in df.columns:
            X_new.at[0, col] = df[col].mean()

    X_scaled = pd_scaler.transform(X_new)
    return X_scaled

# ---------- MAIN PREDICTION FUNCTION ----------
def predict_parkinsons_from_audio(audio_path):
    try:
        # Convert if AAC
        if audio_path.lower().endswith(".aac"):
            wav_path = convert_aac_to_wav(audio_path)
        else:
            wav_path = audio_path

        X_new = prepare_pd_input_from_audio(wav_path)

        pred = pd_model.predict(X_new)[0]
        prob = pd_model.predict_proba(X_new)[0, 1]

        return {
            "prediction": "Parkinson's" if pred == 1 else "Healthy",
            "confidence": float(prob if pred == 1 else 1 - prob)
        }

    except Exception as e:
        return {
            "error": str(e)
        }

result = predict_parkinsons_from_audio("/content/hellosm.aac")
print(result)
