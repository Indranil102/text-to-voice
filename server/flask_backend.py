from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from gtts import gTTS
import os
import uuid
import tempfile
from io import BytesIO
import logging
from werkzeug.utils import secure_filename
import numpy as np
from TTS.api import TTS

app = Flask(__name__)
CORS(app, origins=["http://localhost:5173"])  # Allow Vite frontend

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure temp directory exists
TEMP_DIR = "temp_audio"
if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)

# Initialize Coqui TTS model (XTTS v2 for multilingual voice cloning)
tts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2")

@app.route('/api/text-to-speech', methods=['POST'])
def text_to_speech():
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'Text is required'}), 400
        
        text = data['text'].strip()
        language = data.get('language', 'en')  # Default to English
        
        if not text:
            return jsonify({'error': 'Text cannot be empty'}), 400
        
        # Generate unique filename
        filename = f"tts_{uuid.uuid4().hex}.mp3"
        filepath = os.path.join(TEMP_DIR, filename)
        
        # Create gTTS object and save to file
        tts = gTTS(text=text, lang=language, slow=False)
        tts.save(filepath)
        
        logger.info(f"Generated TTS file: {filename}")
        
        return jsonify({
            'success': True,
            'message': 'Text converted to speech successfully',
            'audio_url': f'/api/audio/{filename}',
            'filename': filename
        })
        
    except Exception as e:
        logger.error(f"Error in text-to-speech conversion: {str(e)}")
        return jsonify({'error': f'Failed to convert text to speech: {str(e)}'}), 500

@app.route('/api/audio/<filename>', methods=['GET'])
def get_audio(filename):
    try:
        filepath = os.path.join(TEMP_DIR, filename)
        
        if not os.path.exists(filepath):
            return jsonify({'error': 'Audio file not found'}), 404
        
        return send_file(filepath, as_attachment=True, download_name=filename)
    
    except Exception as e:
        logger.error(f"Error serving audio file: {str(e)}")
        return jsonify({'error': 'Failed to serve audio file'}), 500

@app.route('/api/languages', methods=['GET'])
def get_supported_languages():
    """Return list of supported languages"""
    languages = {
        'en': 'English',
        'es': 'Spanish',
        'fr': 'French',
        'de': 'German',
        'it': 'Italian',
        'pt': 'Portuguese',
        'ru': 'Russian',
        'ja': 'Japanese',
        'ko': 'Korean',
        'zh': 'Chinese',
        'hi': 'Hindi',
        'ar': 'Arabic'
    }
    return jsonify(languages)

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'service': 'Text-to-Speech API'})

# Cleanup old files (optional)
@app.route('/api/cleanup', methods=['POST'])
def cleanup_files():
    try:
        files_deleted = 0
        for filename in os.listdir(TEMP_DIR):
            if filename.endswith('.mp3'):
                filepath = os.path.join(TEMP_DIR, filename)
                os.remove(filepath)
                files_deleted += 1
        
        return jsonify({
            'success': True,
            'message': f'Cleaned up {files_deleted} audio files'
        })
    
    except Exception as e:
        return jsonify({'error': f'Cleanup failed: {str(e)}'}), 500

@app.route('/api/upload-audio', methods=['POST'])
def upload_audio():
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file part in the request'}), 400
        file = request.files['audio']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        filename = secure_filename(file.filename)
        # Ensure unique filename
        unique_filename = f"user_{uuid.uuid4().hex}_{filename}"
        filepath = os.path.join(TEMP_DIR, unique_filename)
        file.save(filepath)
        # Extract and save speaker embedding
        speaker_embedding = tts_model.get_speaker_embedding(filepath)
        embedding_path = os.path.join(TEMP_DIR, unique_filename + ".npy")
        np.save(embedding_path, speaker_embedding)
        logger.info(f"Received and saved user audio and embedding: {unique_filename}")
        return jsonify({
            'success': True,
            'message': 'Audio uploaded and embedding saved successfully',
            'audio_url': f'/api/audio/{unique_filename}',
            'embedding_filename': unique_filename + ".npy",
            'filename': unique_filename
        })
    except Exception as e:
        logger.error(f"Error uploading audio: {str(e)}")
        return jsonify({'error': f'Failed to upload audio: {str(e)}'}), 500

# New endpoint for custom TTS with user voice
@app.route('/api/custom-tts', methods=['POST'])
def custom_tts():
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        language = data.get('language', 'en')
        embedding_filename = data.get('embedding_filename')  # e.g., user_xxx.wav.npy

        if not text or not embedding_filename:
            return jsonify({'error': 'Text and embedding filename required'}), 400

        embedding_path = os.path.join(TEMP_DIR, embedding_filename)
        if not os.path.exists(embedding_path):
            return jsonify({'error': 'Speaker embedding not found'}), 404

        speaker_embedding = np.load(embedding_path)
        output_filename = f"custom_tts_{uuid.uuid4().hex}.wav"
        output_path = os.path.join(TEMP_DIR, output_filename)

        # Synthesize speech with custom voice
        tts_model.tts_to_file(
            text=text,
            speaker_wav=speaker_embedding,
            language=language,
            file_path=output_path
        )

        return jsonify({
            'success': True,
            'audio_url': f'/api/audio/{output_filename}',
            'filename': output_filename
        })
    except Exception as e:
        logger.error(f"Error in custom TTS: {str(e)}")
        return jsonify({'error': f'Failed to synthesize custom TTS: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)