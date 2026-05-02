import os
import joblib
import pandas as pd
import speech_recognition as sr
from flask import Flask, render_template, request, jsonify
from pydub import AudioSegment

app = Flask(__name__)

MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')

# Load models
disease_model = None
yield_model = None
recommend_model = None

try:
    disease_model = joblib.load(os.path.join(MODELS_DIR, 'disease_model.joblib'))
    yield_model = joblib.load(os.path.join(MODELS_DIR, 'yield_model.joblib'))
    recommend_model = joblib.load(os.path.join(MODELS_DIR, 'recommend_model.joblib'))
except Exception as e:
    print(f"Models not found or failed to load: {e}")

# Routes for Pages
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/disease')
def disease():
    return render_template('disease.html')

@app.route('/yield')
def yield_page():
    return render_template('yield.html')

@app.route('/recommend')
def recommend():
    return render_template('recommend.html')

# API Endpoints
@app.route('/api/voice', methods=['POST'])
def process_voice():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
        
    audio_file = request.files['audio']
    
    # Save the received blob temporarily
    temp_path = "temp_audio.webm"
    wav_path = "temp_audio.wav"
    audio_file.save(temp_path)
    
    try:
        # Convert webm/ogg to wav using pydub
        audio = AudioSegment.from_file(temp_path)
        audio.export(wav_path, format="wav")
        
        # Recognize speech using SpeechRecognition
        recognizer = sr.Recognizer()
        with sr.AudioFile(wav_path) as source:
            audio_data = recognizer.record(source)
            # Use Google Speech Recognition with Hindi language
            text = recognizer.recognize_google(audio_data, language="hi-IN")
            
        return jsonify({'text': text})
        
    except sr.UnknownValueError:
        return jsonify({'error': 'Could not understand audio / आवाज़ समझ में नहीं आई'}), 400
    except sr.RequestError as e:
        return jsonify({'error': f'Speech recognition service error: {e}'}), 500
    except Exception as e:
        return jsonify({'error': f'Processing error: {str(e)}'}), 500
    finally:
        # Cleanup temp files
        if os.path.exists(temp_path):
            os.remove(temp_path)
        if os.path.exists(wav_path):
            os.remove(wav_path)

@app.route('/api/predict/disease', methods=['POST'])
def predict_disease():
    data = request.json
    try:
        temp = float(data.get('temperature', 25))
        humidity = float(data.get('humidity', 50))
        moisture = float(data.get('moisture', 40))
        
        if disease_model:
            df = pd.DataFrame([[temp, humidity, moisture]], columns=['Temperature', 'Humidity', 'Soil Moisture'])
            prediction = disease_model.predict(df)[0]
            return jsonify({'prediction': prediction})
        else:
            return jsonify({'error': 'Model not loaded'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/predict/yield', methods=['POST'])
def predict_yield():
    data = request.json
    try:
        area = float(data.get('area', 1))
        rainfall = float(data.get('rainfall', 200))
        temp = float(data.get('temperature', 25))
        fertilizer = float(data.get('fertilizer', 100))
        
        if yield_model:
            df = pd.DataFrame([[area, rainfall, temp, fertilizer]], columns=['Area', 'Rainfall', 'Temperature', 'Fertilizer'])
            prediction = yield_model.predict(df)[0]
            return jsonify({'prediction': f"{prediction:.2f} टन (Tons)"})
        else:
            return jsonify({'error': 'Model not loaded'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/predict/recommend', methods=['POST'])
def predict_recommend():
    data = request.json
    try:
        n = float(data.get('n', 50))
        p = float(data.get('p', 50))
        k = float(data.get('k', 50))
        ph = float(data.get('ph', 7.0))
        rainfall = float(data.get('rainfall', 200))
        
        if recommend_model:
            df = pd.DataFrame([[n, p, k, ph, rainfall]], columns=['N', 'P', 'K', 'pH', 'Rainfall'])
            prediction = recommend_model.predict(df)[0]
            return jsonify({'prediction': prediction})
        else:
            return jsonify({'error': 'Model not loaded'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
