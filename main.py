"""
main.py

Unified Flask app for Text + Audio emotion/sentiment detection.

Usage:
    python main.py

Expect these files in ./artifacts (optional — app will still start without them):
  - artifacts/clean_text.pkl            # pickled text cleaning function (callable) OR omitted
  - artifacts/vectorizer.pkl
  - artifacts/transform.pkl
  - artifacts/logistic_regression_model.pkl
  - artifacts/sentiment_cnn_model.h5     # keras model for audio (optional)
  - artifacts/le.pkl                     # sklearn LabelEncoder for audio (optional)

Uploads saved to ./uploads (auto-created).
Templates: place your HTML in templates/index.html (I used `index.html` in examples).
"""
import os
import re
import pickle
import string
from pathlib import Path
import warnings
from sklearn.exceptions import InconsistentVersionWarning

# Suppress scikit-learn version warnings
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

import numpy as np
from flask import Flask, request, render_template, send_from_directory, url_for, flash, redirect
from werkzeug.utils import secure_filename

# Optional libs (librosa, tensorflow, sklearn). If missing, app still runs in text-only mode.
try:
    import librosa
except Exception:
    librosa = None

try:
    import tensorflow as tf
except Exception:
    tf = None

try:
    from sklearn.preprocessing import LabelEncoder
except Exception:
    LabelEncoder = None

# -------------------------
# Config / Paths
# -------------------------
BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
UPLOAD_DIR = BASE_DIR / "uploads"
TEMPLATE_NAME = "index.html"  # templates/index.html

ALLOWED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aiff"}
MAX_CONTENT_BYTES = 50 * 1024 * 1024  # 50MB

# ensure directories exist
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
(Path(BASE_DIR) / "templates").mkdir(parents=True, exist_ok=True)

# -------------------------
# App init
# -------------------------
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = str(UPLOAD_DIR)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_BYTES
app.secret_key = os.environ.get("FLASK_SECRET", "dev-secret-change-me")

# -------------------------
# Safe loader helpers
# -------------------------
def load_pickle_safe(path: Path):
    """Load pickle if exists, otherwise return None (and log)."""
    if not path.exists():
        app.logger.info(f"Missing artifact: {path}")
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        app.logger.exception(f"Failed to load pickle {path}: {e}")
        return None

def load_keras_safe(path: Path):
    """Load a Keras model safely if TF is installed."""
    if tf is None:
        app.logger.info("TensorFlow not installed -> audio model disabled.")
        return None
    if not path.exists():
        app.logger.info(f"Audio model not found at {path}")
        return None
    try:
        model = tf.keras.models.load_model(str(path))
        app.logger.info(f"✅ Audio model loaded from {path}")
        return model
    except Exception as e:
        app.logger.exception(f"Failed to load Keras model {path}: {e}")
        return None

# -------------------------
# Load artifacts
# -------------------------
# As requested: use artifacts/clean_text.pkl with raw path string
text_clean_function = load_pickle_safe(ARTIFACTS_DIR / "clean_text.pkl")  # may be a callable

vectorizer = load_pickle_safe(ARTIFACTS_DIR / "vectorizer.pkl")
tfidfconverter = load_pickle_safe(ARTIFACTS_DIR / "transform.pkl")
pickled_model = load_pickle_safe(ARTIFACTS_DIR / "logistic_regression_model.pkl")

audio_model = load_keras_safe(ARTIFACTS_DIR / "sentiment_cnn_model.h5")
le = load_pickle_safe(ARTIFACTS_DIR / "le.pkl")  # LabelEncoder (optional)

# -------------------------
# Model Inspector
# -------------------------
def inspect_model_structure():
    """Print model architecture to understand input requirements"""
    if audio_model is not None:
        print("\n" + "="*50)
        print("AUDIO MODEL STRUCTURE")
        print("="*50)
        
        # Get input shape
        input_shape = audio_model.input_shape
        print(f"📊 Input shape: {input_shape}")
        
        print("\n🔍 Layer Details:")
        for i, layer in enumerate(audio_model.layers):
            layer_info = f"  Layer {i}: {layer.name}"
            if hasattr(layer, 'input_shape'):
                layer_info += f" | Input: {layer.input_shape}"
            if hasattr(layer, 'output_shape'):
                layer_info += f" | Output: {layer.output_shape}"
            print(layer_info)
        
        print("="*50 + "\n")
        return input_shape
    return None

# -------------------------
# Text preprocessing pipeline
# -------------------------
# Try to use NLTK stopwords if available (optional)
try:
    import nltk
    from nltk.corpus import stopwords
    _ = stopwords.words("english")
    NLTK_STOPWORDS_AVAILABLE = True
except Exception:
    NLTK_STOPWORDS_AVAILABLE = False
    stopwords = set()

def fallback_clean_text(text: str) -> str:
    """Robust fallback cleaning: lowercase, remove urls, mentions, punctuation, extra spaces, optional stopwords."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"@([A-Za-z0-9_]+)", r"\1", text)   # keep the handle word, remove '@'
    text = re.sub(r"#([A-Za-z0-9_]+)", r"\1", text)   # keep the hashtag word
    text = re.sub(r"\brt\b", "", text)                # remove RT token
    # remove unicode escape sequences like \u1234
    text = re.sub(r"\\u[0-9A-Fa-f]{4,6}", "", text)
    # remove punctuation (keep apostrophe optionally)
    text = text.translate(str.maketrans("", "", string.punctuation.replace("'", "")))
    text = re.sub(r"\s+", " ", text).strip()
    if NLTK_STOPWORDS_AVAILABLE:
        words = [w for w in text.split() if w not in stopwords.words("english")]
        text = " ".join(words)
    return text

# Compose pipeline: first try user pickled function, then fallback
if text_clean_function and callable(text_clean_function):
    def clean_text_pipeline(text: str) -> str:
        try:
            t = text_clean_function(text)
            return fallback_clean_text(t)
        except Exception:
            return fallback_clean_text(text)
else:
    clean_text_pipeline = fallback_clean_text

# -------------------------
# Audio helpers
# -------------------------
def allowed_audio(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_AUDIO_EXTENSIONS

def extract_mfcc_features(file_path: str):
    """Extract MFCC features with 3-10 second audio limit"""
    if librosa is None:
        raise RuntimeError("librosa not installed; cannot extract audio features.")
    
    try:
        # Load audio with 10 second limit (or use full if shorter)
        audio, sample_rate = librosa.load(
            file_path, 
            res_type='kaiser_fast', 
            duration=10.0,  # Max 10 seconds
            sr=16000  # Optional: resample to consistent rate
        )
        
        duration = len(audio) / sample_rate
        app.logger.info(f"📊 Processed audio: {duration:.2f} seconds, Sample rate: {sample_rate}Hz")
        
        # Ensure we have at least 1 second of audio
        if duration < 1.0:
            app.logger.warning(f"Audio too short ({duration:.2f}s). Using full length.")
            audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        
        # Calculate MFCC features
        mfccs = librosa.feature.mfcc(
            y=audio, 
            sr=sample_rate, 
            n_mfcc=40,
            n_fft=2048,
            hop_length=512
        )
        
        # Take mean across time frames
        features = np.mean(mfccs.T, axis=0)
        
        # Log feature extraction details
        app.logger.info(f"📊 MFCC shape before mean: {mfccs.shape}")
        app.logger.info(f"📊 Final feature shape: {features.shape}")
        
        return features
        
    except Exception as e:
        app.logger.exception(f"Error extracting MFCC from {file_path}: {e}")
        raise

def predict_audio_sentiment(file_path: str):
    """
    Predict using loaded Keras model and LabelEncoder.
    Returns label string and confidence or None.
    """
    if audio_model is None:
        app.logger.info("Audio model not loaded -> returning None for audio prediction.")
        return None, None
    
    try:
        # Use the SAME feature extraction as the working code
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        feature = np.mean(mfccs.T, axis=0)  # This gives shape (40,)
        
        app.logger.info(f"📊 Extracted feature shape: {feature.shape}")
        app.logger.info(f"🤖 Model input shape: {audio_model.input_shape}")
        
        # Reshape EXACTLY as the working code does
        # Model expects: (batch_size, 40, 1, 1)
        feature = feature.reshape(1, 40, 1, 1)
        
        app.logger.info(f"🔄 Reshaped feature for model: {feature.shape}")
        
        # Predict
        preds = audio_model.predict(feature, verbose=0)
        
        predicted_index = int(np.argmax(preds, axis=1)[0])
        confidence = float(np.max(preds))
        
        app.logger.info(f"🎯 Predicted index: {predicted_index}, Confidence: {confidence:.4f}")
        
        # Decode label
        if le is not None and hasattr(le, "inverse_transform"):
            try:
                label = le.inverse_transform([predicted_index])[0]
            except Exception as e:
                app.logger.warning(f"Failed to inverse transform label: {e}")
                label = str(predicted_index)
        else:
            label = str(predicted_index)
        
        return label, confidence
        
    except Exception as e:
        app.logger.exception("Failed to predict audio sentiment: %s", e)
        return None, None

# -------------------------
# Routes
# -------------------------
@app.route("/", methods=["GET"])
def home():
    # initial render; template should gracefully handle None values
    return render_template(TEMPLATE_NAME,
                           predicted_emotion_text=None, predicted_confidence_text=None,
                           predicted_explanation_text=None,
                           predicted_emotion_audio=None, predicted_confidence_audio=None,
                           audio_path=None, audio_transcript=None)

@app.route("/predict_text", methods=["POST"])
def predict_text():
    sentence = request.form.get("sentence", "")
    if not sentence:
        flash("Please enter text to analyze.", "warning")
        return redirect(url_for("home"))

    cleaned_text = clean_text_pipeline(sentence)

    if vectorizer is None or tfidfconverter is None or pickled_model is None:
        flash("Text model/artifacts missing. Put vectorizer.pkl, transform.pkl and logistic_regression_model.pkl into artifacts/", "danger")
        return render_template(TEMPLATE_NAME,
                               predicted_emotion_text=None, predicted_emotion_audio=None, audio_path=None)

    try:
        features = vectorizer.transform([cleaned_text])
        # some pipelines use fit_transform by mistake — try transform then fallback to fit_transform
        try:
            tfidf_features = tfidfconverter.transform(features).toarray()
        except Exception:
            tfidf_features = tfidfconverter.fit_transform(features).toarray()

        preds = pickled_model.predict(tfidf_features)
        pred_label = preds[0] if hasattr(preds, "__len__") else str(preds)
        confidence = None
        if hasattr(pickled_model, "predict_proba"):
            try:
                probs = pickled_model.predict_proba(tfidf_features)
                confidence = float(np.max(probs))
            except Exception:
                confidence = None

    except Exception as e:
        app.logger.exception("Text prediction failed: %s", e)
        flash("Text prediction error. Check server logs.", "danger")
        return render_template(TEMPLATE_NAME,
                               predicted_emotion_text=None, predicted_emotion_audio=None, audio_path=None)

    return render_template(TEMPLATE_NAME,
                           predicted_emotion_text=str(pred_label).upper(),
                           predicted_confidence_text=confidence,
                           predicted_explanation_text=None,
                           predicted_emotion_audio=None,
                           audio_path=None)

@app.route("/predict_audio", methods=["POST"])
def predict_audio():
    if "audio" not in request.files:
        flash("No audio part found in request.", "warning")
        return redirect(url_for("home"))

    f = request.files["audio"]
    if f.filename == "":
        flash("No audio file selected.", "warning")
        return redirect(url_for("home"))

    if not allowed_audio(f.filename):
        flash("This file type is not allowed. Accepts wav, mp3, m4a, flac, ogg, aiff.", "warning")
        return redirect(url_for("home"))

    filename = secure_filename(f.filename)
    save_path = Path(app.config["UPLOAD_FOLDER"]) / filename
    try:
        f.save(str(save_path))
    except Exception as e:
        app.logger.exception("Failed to save uploaded file: %s", e)
        flash("Failed to save uploaded file.", "danger")
        return redirect(url_for("home"))

    predicted_label, confidence = predict_audio_sentiment(str(save_path))
    if predicted_label is None:
        flash("Audio model not available or prediction failed. Check server logs.", "danger")
        return render_template(TEMPLATE_NAME,
                               predicted_emotion_audio=None,
                               predicted_confidence_audio=None,
                               audio_path=str(save_path),
                               predicted_emotion_text=None)

    return render_template(TEMPLATE_NAME,
                           predicted_emotion_audio=str(predicted_label).upper(),
                           predicted_confidence_audio=confidence,
                           audio_path=str(save_path),
                           predicted_emotion_text=None)

@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    # Serve uploaded audio for playback in browser
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename, as_attachment=False)

# -------------------------
# Debug route to check model structure
# -------------------------
@app.route("/debug/model", methods=["GET"])
def debug_model():
    """Debug endpoint to check model structure"""
    if audio_model is None:
        return "Audio model not loaded", 404
    
    result = ["<h1>Model Structure</h1>"]
    result.append(f"<p>Input shape: {audio_model.input_shape}</p>")
    result.append("<h2>Layers:</h2><ul>")
    
    for i, layer in enumerate(audio_model.layers):
        layer_info = f"<li>Layer {i}: {layer.name}"
        if hasattr(layer, 'input_shape'):
            layer_info += f", Input: {layer.input_shape}"
        if hasattr(layer, 'output_shape'):
            layer_info += f", Output: {layer.output_shape}"
        if hasattr(layer, 'units'):
            layer_info += f", Units: {layer.units}"
        layer_info += "</li>"
        result.append(layer_info)
    
    result.append("</ul>")
    return "".join(result)

# -------------------------
# CLI entry
# -------------------------
if __name__ == "__main__":
    # Startup banner
    print("\n" + "="*50)
    print("🎭 MULTI-MODAL EMOTION DETECTION APP")
    print("="*50)
    
    # Check loaded models
    print(f"\n✅ Text model loaded: {pickled_model is not None}")
    print(f"✅ Audio model loaded: {audio_model is not None}")
    print(f"✅ LabelEncoder loaded: {le is not None}")
    print(f"✅ Clean function loaded: {text_clean_function is not None}")
    
    # Inspect audio model structure
    if audio_model is not None:
        inspect_model_structure()
        print("🎯 Audio model expects: (batch_size, 40, 1, 1)")
        print("   - 40 MFCC features")
        print("   - 1 time step (mean across time)")
        print("   - 1 channel")
    
    print("\n🌐 Server starting on http://localhost:5000")
    print("📁 Uploads folder:", app.config["UPLOAD_FOLDER"])
    print("="*50 + "\n")
    
    # For dev only: Debug True. In production: set debug=False and use a WSGI server.
    app.run(debug=True, port=5000)