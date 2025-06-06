from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from gtts import gTTS
import os
import uuid
import tempfile
from io import BytesIO
import logging

app = Flask(__name__)
CORS(app, origins=["http://localhost:5173"])  # Allow Vite frontend

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure temp directory exists
TEMP_DIR = "temp_audio"
if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)

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

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)