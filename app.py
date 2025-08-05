from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import base64
import tempfile
import logging
from datetime import datetime
import torch
import torchaudio
from transformers import AutoProcessor, BarkModel
import scipy.io.wavfile as wavfile
import numpy as np
import soundfile as sf

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configure CORS
CORS(app, origins=[
    'https://mythiq-ui-production.up.railway.app',
    'http://localhost:5173',
    'http://localhost:3000'
])

# Global variables for models (lazy loading)
bark_model = None
bark_processor = None

def load_bark_model():
    """Load Bark TTS model (lazy loading)"""
    global bark_model, bark_processor
    if bark_model is None:
        try:
            logger.info("Loading Bark TTS model...")
            bark_processor = AutoProcessor.from_pretrained("suno/bark-small")
            bark_model = BarkModel.from_pretrained("suno/bark-small")
            
            # Move to GPU if available
            device = "cuda" if torch.cuda.is_available() else "cpu"
            bark_model = bark_model.to(device)
            logger.info(f"Bark model loaded successfully on {device}")
        except Exception as e:
            logger.error(f"Failed to load Bark model: {str(e)}")
            raise e
    return bark_model, bark_processor

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        return jsonify({
            "status": "online",
            "service": "mythiq-audio-creator",
            "timestamp": datetime.now().isoformat(),
            "models_loaded": {
                "bark": bark_model is not None,
            },
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "cuda_available": torch.cuda.is_available(),
            "message": "Audio generation service ready",
            "features": [
                "Speech generation with Bark AI",
                "Music generation (procedural)",
                "Multiple voice presets",
                "Real-time audio generation"
            ]
        }), 200
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

@app.route('/voice-presets', methods=['GET'])
def get_voice_presets():
    """Get available voice presets"""
    try:
        presets = [
            {
                "id": "v2/en_speaker_0",
                "name": "English Speaker 0 (male)",
                "language": "en",
                "gender": "male",
                "description": "Clear male voice"
            },
            {
                "id": "v2/en_speaker_1", 
                "name": "English Speaker 1 (female)",
                "language": "en",
                "gender": "female",
                "description": "Clear female voice"
            },
            {
                "id": "v2/en_speaker_6",
                "name": "English Speaker 6 (male)",
                "language": "en", 
                "gender": "male",
                "description": "Deep male voice"
            }
        ]
        
        return jsonify({
            "success": True,
            "presets": presets,
            "total": len(presets)
        }), 200
        
    except Exception as e:
        logger.error(f"Voice presets error: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/generate-speech', methods=['POST'])
def generate_speech():
    """Generate speech from text using Bark TTS"""
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({
                "success": False,
                "error": "Missing 'text' parameter"
            }), 400
        
        text = data['text']
        voice_preset = data.get('voice_preset', 'v2/en_speaker_0')
        
        if len(text) > 1000:
            return jsonify({
                "success": False,
                "error": "Text too long. Maximum 1000 characters."
            }), 400
        
        logger.info(f"Generating speech for text: {text[:50]}...")
        
        # Load model if not already loaded
        model, processor = load_bark_model()
        
        # Prepare input with voice preset
        inputs = processor(text, voice_preset=voice_preset, return_tensors="pt")
        
        # Move inputs to same device as model
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate speech
        with torch.no_grad():
            audio_array = model.generate(**inputs)
        
        # Convert to numpy and ensure correct format
        audio_array = audio_array.cpu().numpy().squeeze()
        
        # Normalize audio
        audio_array = audio_array / np.max(np.abs(audio_array))
        
        # Convert to 16-bit PCM
        audio_int16 = (audio_array * 32767).astype(np.int16)
        
        # Create WAV file in memory
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            wavfile.write(tmp_file.name, 24000, audio_int16)  # Bark uses 24kHz
            
            # Read file and encode as base64
            with open(tmp_file.name, 'rb') as f:
                audio_data = f.read()
                audio_base64 = base64.b64encode(audio_data).decode('utf-8')
            
            # Clean up temp file
            os.unlink(tmp_file.name)
        
        # Calculate duration
        duration = len(audio_int16) / 24000  # 24kHz sample rate
        
        return jsonify({
            "success": True,
            "message": "🗣️ Speech generated successfully!",
            "audio_data": f"data:audio/wav;base64,{audio_base64}",
            "generation_info": {
                "text": text,
                "voice_preset": voice_preset,
                "duration": round(duration, 2),
                "sample_rate": 24000,
                "format": "wav",
                "file_size": f"{len(audio_data) / 1024:.1f} KB"
            },
            "timestamp": datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Speech generation error: {str(e)}")
        return jsonify({
            "success": False,
            "error": f"Speech generation failed: {str(e)}"
        }), 500

@app.route('/generate-music', methods=['POST'])
def generate_music():
    """Generate music from text prompt"""
    try:
        data = request.get_json()
        
        if not data or 'prompt' not in data:
            return jsonify({
                "success": False,
                "error": "Missing 'prompt' parameter"
            }), 400
        
        prompt = data['prompt']
        duration = data.get('duration', 10)  # Default 10 seconds
        
        if duration > 30:
            duration = 30  # Cap at 30 seconds for free tier
        
        logger.info(f"Generating music for prompt: {prompt}")
        
        # Generate procedural music based on prompt
        sample_rate = 44100
        samples = int(sample_rate * duration)
        
        # Create a simple melody with multiple frequencies
        t = np.linspace(0, duration, samples)
        
        # Analyze prompt for music style
        prompt_lower = prompt.lower()
        if 'electronic' in prompt_lower or 'dance' in prompt_lower:
            # Electronic/Dance music
            frequencies = [440, 554, 659, 784]  # A major chord
            bass_freq = 110  # Bass line
        elif 'classical' in prompt_lower or 'piano' in prompt_lower:
            # Classical music
            frequencies = [261, 329, 392, 523]  # C major chord
            bass_freq = 130
        elif 'rock' in prompt_lower or 'guitar' in prompt_lower:
            # Rock music
            frequencies = [329, 415, 493, 659]  # E major chord
            bass_freq = 82
        else:
            # Default pleasant melody
            frequencies = [440, 554, 659, 784]  # A major chord
            bass_freq = 110
        
        audio = np.zeros(samples)
        
        # Add melody
        for i, freq in enumerate(frequencies):
            amplitude = 0.2 / (i + 1)
            audio += amplitude * np.sin(2 * np.pi * freq * t) * np.exp(-t * 0.3)
        
        # Add bass line
        audio += 0.3 * np.sin(2 * np.pi * bass_freq * t) * np.exp(-t * 0.2)
        
        # Add rhythm based on style
        if 'upbeat' in prompt_lower or 'dance' in prompt_lower:
            beat_freq = 2.5  # Fast beat
        elif 'slow' in prompt_lower or 'calm' in prompt_lower:
            beat_freq = 1.0  # Slow beat
        else:
            beat_freq = 2.0  # Medium beat
            
        beat_envelope = 0.6 + 0.4 * np.sin(2 * np.pi * beat_freq * t)
        audio *= beat_envelope
        
        # Add some harmonics for richness
        audio += 0.1 * np.sin(2 * np.pi * frequencies[0] * 2 * t) * np.exp(-t * 0.5)
        
        # Normalize
        audio = audio / np.max(np.abs(audio)) * 0.8
        
        # Convert to 16-bit PCM
        audio_int16 = (audio * 32767).astype(np.int16)
        
        # Create WAV file in memory
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            wavfile.write(tmp_file.name, sample_rate, audio_int16)
            
            # Read file and encode as base64
            with open(tmp_file.name, 'rb') as f:
                audio_data = f.read()
                audio_base64 = base64.b64encode(audio_data).decode('utf-8')
            
            # Clean up temp file
            os.unlink(tmp_file.name)
        
        return jsonify({
            "success": True,
            "message": "🎶 Music generated successfully!",
            "audio_data": f"data:audio/wav;base64,{audio_base64}",
            "generation_info": {
                "prompt": prompt,
                "duration": duration,
                "sample_rate": sample_rate,
                "format": "wav",
                "file_size": f"{len(audio_data) / 1024:.1f} KB",
                "style": "Procedural music generation based on prompt analysis"
            },
            "timestamp": datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Music generation error: {str(e)}")
        return jsonify({
            "success": False,
            "error": f"Music generation failed: {str(e)}"
        }), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "success": False,
        "error": "Endpoint not found"
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        "success": False,
        "error": "Internal server error"
    }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"Starting Mythiq Audio Creator on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)
