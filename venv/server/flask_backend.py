''' from flask import Flask, request, jsonify, send_file
from flask_get_ollama_responseors import CORS
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
    app.run(debug=True, host='0.0.0.0', port=5001)'''


from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from gtts import gTTS
import os
import uuid
import tempfile
from io import BytesIO
import logging
import requests
import json

app = Flask(__name__)
CORS(app, origins=["http://localhost:5173"])  # Allow Vite frontend

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure temp directory exists
TEMP_DIR = "temp_audio"
if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)

# Ollama configuration
OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "llama2:latest"
  # Change as needed

def get_ollama_response(prompt, model=DEFAULT_MODEL):
    """Get response from Ollama API"""
    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False
            },
            timeout=60
        )
        if response.status_code == 200:
            return response.json().get("response")
        else:
            logger.error(f"Ollama API error: {response.status_code} - {response.text}")
            return None
    except requests.RequestException as e:
        logger.error(f"Error connecting to Ollama: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error with Ollama: {e}")
        return None

def check_ollama_connection():
    """Check if Ollama is running and accessible"""
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        if response.status_code == 200:
            return True
        logger.error(f"Ollama connection check failed: {response.status_code} - {response.text}")
        return False
    except requests.RequestException as e:
        logger.error(f"Error connecting to Ollama: {e}")
        return False

def get_available_models():
    """Get list of available Ollama models"""
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=10)
        if response.status_code == 200:
            data = response.json()
            models = data.get("models", [])
            return [model.get("name").split(":")[0] for model in models if "name" in model]
        else:
            logger.error(f"Failed to get models: {response.status_code} - {response.text}")
            return []
    except requests.RequestException as e:
        logger.error(f"Error fetching models from Ollama: {e}")
        return []


# Original TTS endpoint
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

# New endpoint: Chat with Ollama (text only)
@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        
        if not data or 'question' not in data:
            return jsonify({'error': 'Question is required'}), 400
        
        question = data['question'].strip()
        model = data.get('model', DEFAULT_MODEL)
        
        if not question:
            return jsonify({'error': 'Question cannot be empty'}), 400
        
        # Check Ollama connection
        if not check_ollama_connection():
            return jsonify({'error': 'Ollama service is not available. Please make sure Ollama is running.'}), 503
        
        # Get response from Ollama
        ollama_response = get_ollama_response(question, model)
        
        if ollama_response is None:
            return jsonify({'error': 'Failed to get response from Ollama'}), 500
        
        return jsonify({
            'success': True,
            'question': question,
            'response': ollama_response,
            'model': model
        })
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        return jsonify({'error': f'Chat failed: {str(e)}'}), 500

# New endpoint: Chat with Ollama and convert to speech
@app.route('/api/chat-and-speak', methods=['POST'])
def chat_and_speak():
    try:
        data = request.get_json()
        
        if not data or 'question' not in data:
            return jsonify({'error': 'Question is required'}), 400
        
        question = data['question'].strip()
        language = data.get('language', 'en')
        model = data.get('model', DEFAULT_MODEL)
        
        if not question:
            return jsonify({'error': 'Question cannot be empty'}), 400
        
        # Check Ollama connection
        if not check_ollama_connection():
            return jsonify({'error': 'Ollama service is not available. Please make sure Ollama is running.'}), 503
        
        # Get response from Ollama
        ollama_response = get_ollama_response(question, model)
        
        if ollama_response is None:
            return jsonify({'error': 'Failed to get response from Ollama'}), 500
        
        # Convert response to speech
        filename = f"tts_{uuid.uuid4().hex}.mp3"
        filepath = os.path.join(TEMP_DIR, filename)
        
        tts = gTTS(text=ollama_response, lang=language, slow=False)
        tts.save(filepath)
        
        logger.info(f"Generated chat response and TTS file: {filename}")
        
        return jsonify({
            'success': True,
            'question': question,
            'response': ollama_response,
            'audio_url': f'/api/audio/{filename}',
            'filename': filename,
            'model': model
        })
        
    except Exception as e:
        logger.error(f"Error in chat-and-speak endpoint: {str(e)}")
        return jsonify({'error': f'Chat and speak failed: {str(e)}'}), 500

# Get available Ollama models
@app.route('/api/models', methods=['GET'])
def get_models():
    """Get list of available Ollama models"""
    try:
        if not check_ollama_connection():
            return jsonify({'error': 'Ollama service is not available'}), 503
        
        models = get_available_models()
        return jsonify({
            'success': True,
            'models': models,
            'default_model': DEFAULT_MODEL
        })
        
    except Exception as e:
        logger.error(f"Error getting models: {str(e)}")
        return jsonify({'error': 'Failed to get models'}), 500

# Check Ollama status
@app.route('/api/ollama-status', methods=['GET'])
def ollama_status():
    """Check Ollama connection status"""
    is_connected = check_ollama_connection()
    models = get_available_models() if is_connected else []
    
    return jsonify({
        'connected': is_connected,
        'url': OLLAMA_URL,
        'models': models,
        'default_model': DEFAULT_MODEL
    })

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
    ollama_connected = check_ollama_connection()
    
    return jsonify({
        'status': 'healthy',
        'service': 'Text-to-Speech + Ollama API',
        'ollama_connected': ollama_connected,
        'ollama_url': OLLAMA_URL
    })

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
    print("Starting Flask server with Ollama integration...")
    print(f"Ollama URL: {OLLAMA_URL}")
    print(f"Default Model: {DEFAULT_MODEL}")
    
    # Check Ollama connection on startup
    if check_ollama_connection():
        print("✅ Ollama connection successful!")
        models = get_available_models()
        print(f"Available models: {models}")
    else:
        print("⚠️ Warning: Could not connect to Ollama. Make sure Ollama is running.")
    
    app.run(debug=True, host='0.0.0.0', port=5001)