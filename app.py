from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import tempfile
import logging
import traceback
from datetime import datetime
import torch
import torchaudio
from transformers import BarkModel, BarkProcessor
from audiocraft.models import MusicGen
import scipy.io.wavfile
import numpy as np
import io
import base64

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configure CORS to allow requests from your frontend
CORS(app, origins=[
    'https://mythiq-ui-production.up.railway.app',
    'http://localhost:5173',
    'http://localhost:3000'
])

# Global variables for models (lazy loading)
bark_model = None
bark_processor = None
musicgen_model = None

def load_bark_model():
    """Load Bark model for speech generation"""
    global bark_model, bark_processor
    if bark_model is None:
        try:
            logger.info("Loading Bark model...")
            bark_processor = BarkProcessor.from_pretrained("suno/bark-small")
            bark_model = BarkModel.from_pretrained("suno/bark-small")
            
            # Move to GPU if available
            if torch.cuda.is_available():
                bark_model = bark_model.to("cuda")
                logger.info("Bark model loaded on GPU")
            else:
                logger.info("Bark model loaded on CPU")
                
        except Exception as e:
            logger.error(f"Error loading Bark model: {str(e)}")
            raise e
    
    return bark_model, bark_processor

def load_musicgen_model():
    """Load MusicGen model for music generation"""
    global musicgen_model
    if musicgen_model is None:
        try:
            logger.info("Loading MusicGen model...")
            musicgen_model = MusicGen.get_pretrained('facebook/musicgen-small')
            logger.info("MusicGen model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading MusicGen model: {str(e)}")
            raise e
    
    return musicgen_model

def generate_speech_with_bark(text, voice_preset="v2/en_speaker_6"):
    """Generate speech using Bark model"""
    try:
        model, processor = load_bark_model()
        
        # Process the text
        inputs = processor(text, voice_preset=voice_preset, return_tensors="pt")
        
        # Move inputs to same device as model
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        
        # Generate audio
        with torch.no_grad():
            audio_array = model.generate(**inputs)
        
        # Convert to numpy and ensure correct format
        if torch.cuda.is_available():
            audio_array = audio_array.cpu()
        
        audio_array = audio_array.numpy().squeeze()
        
        # Normalize audio
        audio_array = audio_array / np.max(np.abs(audio_array))
        
        # Convert to 16-bit PCM
        audio_array = (audio_array * 32767).astype(np.int16)
        
        return audio_array, 24000  # Bark uses 24kHz sample rate
        
    except Exception as e:
        logger.error(f"Error generating speech: {str(e)}")
        raise e

def generate_music_with_musicgen(prompt, duration=10):
    """Generate music using MusicGen model"""
    try:
        model = load_musicgen_model()
        
        # Set generation parameters
        model.set_generation_params(duration=duration)
        
        # Generate music
        descriptions = [prompt]
        wav = model.generate(descriptions)
        
        # Convert to numpy array
        audio_array = wav[0].cpu().numpy().squeeze()
        
        # Normalize audio
        audio_array = audio_array / np.max(np.abs(audio_array))
        
        # Convert to 16-bit PCM
        audio_array = (audio_array * 32767).astype(np.int16)
        
        return audio_array, model.sample_rate
        
    except Exception as e:
        logger.error(f"Error generating music: {str(e)}")
        raise e

def audio_to_base64(audio_array, sample_rate):
    """Convert audio array to base64 encoded WAV"""
    try:
        # Create a BytesIO buffer
        buffer = io.BytesIO()
        
        # Write WAV data to buffer
        scipy.io.wavfile.write(buffer, sample_rate, audio_array)
        
        # Get the WAV data
        wav_data = buffer.getvalue()
        
        # Encode to base64
        base64_audio = base64.b64encode(wav_data).decode('utf-8')
        
        return base64_audio
        
    except Exception as e:
        logger.error(f"Error converting audio to base64: {str(e)}")
        raise e

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        # Check if models can be loaded
        bark_status = "available"
        musicgen_status = "available"
        
        try:
            load_bark_model()
        except:
            bark_status = "error"
            
        try:
            load_musicgen_model()
        except:
            musicgen_status = "error"
        
        return jsonify({
            "status": "online",
            "service": "mythiq-audio-creator",
            "timestamp": datetime.now().isoformat(),
            "models": {
                "bark": bark_status,
                "musicgen": musicgen_status
            },
            "gpu_available": torch.cuda.is_available(),
            "features": [
                "Speech generation with Bark",
                "Music generation with MusicGen",
                "Multiple voice presets",
                "Customizable audio duration"
            ]
        })
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

@app.route('/generate-speech', methods=['POST'])
def generate_speech():
    """Generate speech from text using Bark"""
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({
                "success": False,
                "error": "Missing 'text' parameter"
            }), 400
        
        text = data['text']
        voice_preset = data.get('voice_preset', 'v2/en_speaker_6')
        
        # Validate text length
        if len(text) > 1000:
            return jsonify({
                "success": False,
                "error": "Text too long. Maximum 1000 characters."
            }), 400
        
        logger.info(f"Generating speech for text: {text[:50]}...")
        
        # Generate speech
        audio_array, sample_rate = generate_speech_with_bark(text, voice_preset)
        
        # Convert to base64
        audio_base64 = audio_to_base64(audio_array, sample_rate)
        
        return jsonify({
            "success": True,
            "audio_data": f"data:audio/wav;base64,{audio_base64}",
            "sample_rate": sample_rate,
            "duration": len(audio_array) / sample_rate,
            "voice_preset": voice_preset,
            "text": text
        })
        
    except Exception as e:
        logger.error(f"Speech generation error: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            "success": False,
            "error": f"Speech generation failed: {str(e)}"
        }), 500

@app.route('/generate-music', methods=['POST'])
def generate_music():
    """Generate music from text prompt using MusicGen"""
    try:
        data = request.get_json()
        
        if not data or 'prompt' not in data:
            return jsonify({
                "success": False,
                "error": "Missing 'prompt' parameter"
            }), 400
        
        prompt = data['prompt']
        duration = min(data.get('duration', 10), 30)  # Max 30 seconds
        
        logger.info(f"Generating music for prompt: {prompt}")
        
        # Generate music
        audio_array, sample_rate = generate_music_with_musicgen(prompt, duration)
        
        # Convert to base64
        audio_base64 = audio_to_base64(audio_array, sample_rate)
        
        return jsonify({
            "success": True,
            "audio_data": f"data:audio/wav;base64,{audio_base64}",
            "sample_rate": sample_rate,
            "duration": duration,
            "prompt": prompt
        })
        
    except Exception as e:
        logger.error(f"Music generation error: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            "success": False,
            "error": f"Music generation failed: {str(e)}"
        }), 500

@app.route('/voice-presets', methods=['GET'])
def get_voice_presets():
    """Get available voice presets for Bark"""
    presets = {
        "english": [
            {"id": "v2/en_speaker_0", "name": "English Speaker 0", "gender": "male"},
            {"id": "v2/en_speaker_1", "name": "English Speaker 1", "gender": "female"},
            {"id": "v2/en_speaker_2", "name": "English Speaker 2", "gender": "male"},
            {"id": "v2/en_speaker_3", "name": "English Speaker 3", "gender": "female"},
            {"id": "v2/en_speaker_4", "name": "English Speaker 4", "gender": "male"},
            {"id": "v2/en_speaker_5", "name": "English Speaker 5", "gender": "female"},
            {"id": "v2/en_speaker_6", "name": "English Speaker 6", "gender": "male"},
            {"id": "v2/en_speaker_7", "name": "English Speaker 7", "gender": "female"},
            {"id": "v2/en_speaker_8", "name": "English Speaker 8", "gender": "male"},
            {"id": "v2/en_speaker_9", "name": "English Speaker 9", "gender": "female"}
        ],
        "multilingual": [
            {"id": "v2/zh_speaker_0", "name": "Chinese Speaker 0", "language": "Chinese"},
            {"id": "v2/zh_speaker_1", "name": "Chinese Speaker 1", "language": "Chinese"},
            {"id": "v2/fr_speaker_0", "name": "French Speaker 0", "language": "French"},
            {"id": "v2/fr_speaker_1", "name": "French Speaker 1", "language": "French"},
            {"id": "v2/de_speaker_0", "name": "German Speaker 0", "language": "German"},
            {"id": "v2/de_speaker_1", "name": "German Speaker 1", "language": "German"},
            {"id": "v2/es_speaker_0", "name": "Spanish Speaker 0", "language": "Spanish"},
            {"id": "v2/es_speaker_1", "name": "Spanish Speaker 1", "language": "Spanish"}
        ]
    }
    
    return jsonify({
        "success": True,
        "presets": presets
    })

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
    app.run(host='0.0.0.0', port=port, debug=False)
