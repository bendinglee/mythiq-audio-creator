from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import base64
import tempfile
import logging
import traceback
from datetime import datetime
import torch
import torchaudio
from transformers import AutoProcessor, BarkModel
import scipy.io.wavfile as wavfile
import numpy as np
import soundfile as sf
import warnings

# Suppress warnings for cleaner logs
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
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
model_loading = False

def load_bark_model():
    """Load Bark TTS model with proper configuration"""
    global bark_model, bark_processor, model_loading
    
    if bark_model is not None:
        return bark_model, bark_processor
    
    if model_loading:
        # Another request is already loading the model
        import time
        for _ in range(60):  # Wait up to 60 seconds
            time.sleep(1)
            if bark_model is not None:
                return bark_model, bark_processor
        raise Exception("Model loading timeout")
    
    try:
        model_loading = True
        logger.info("Loading Bark TTS model...")
        
        # Load processor and model with proper configuration
        bark_processor = AutoProcessor.from_pretrained(
            "suno/bark-small",
            torch_dtype=torch.float32
        )
        bark_model = BarkModel.from_pretrained(
            "suno/bark-small",
            torch_dtype=torch.float32
        )
        
        # Move to GPU if available, otherwise CPU
        device = "cuda" if torch.cuda.is_available() else "cpu"
        bark_model = bark_model.to(device)
        
        # Set model to evaluation mode
        bark_model.eval()
        
        logger.info(f"Bark model loaded successfully on {device}")
        model_loading = False
        return bark_model, bark_processor
        
    except Exception as e:
        model_loading = False
        logger.error(f"Failed to load Bark model: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise e

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
            ],
            "version": "2.0-fixed"
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
    """Generate speech from text using Bark TTS with proper error handling"""
    try:
        # Validate request
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({
                "success": False,
                "error": "Missing 'text' parameter"
            }), 400
        
        text = data['text'].strip()
        voice_preset = data.get('voice_preset', 'v2/en_speaker_0')
        
        # Validate input
        if not text:
            return jsonify({
                "success": False,
                "error": "Text cannot be empty"
            }), 400
            
        if len(text) > 500:  # Reduced limit for stability
            return jsonify({
                "success": False,
                "error": "Text too long. Maximum 500 characters."
            }), 400
        
        logger.info(f"Generating speech for text: {text[:50]}... (voice: {voice_preset})")
        
        # Load model if not already loaded
        try:
            model, processor = load_bark_model()
        except Exception as e:
            logger.error(f"Model loading failed: {str(e)}")
            return jsonify({
                "success": False,
                "error": f"Model loading failed: {str(e)}"
            }), 500
        
        # Prepare input with proper configuration
        try:
            # Create inputs with attention mask
            inputs = processor(
                text, 
                voice_preset=voice_preset, 
                return_tensors="pt",
                return_attention_mask=True,
                padding=True,
                truncation=True,
                max_length=256
            )
            
            # Move inputs to same device as model
            device = next(model.parameters()).device
            inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}
            
            logger.info("Inputs prepared successfully")
            
        except Exception as e:
            logger.error(f"Input preparation failed: {str(e)}")
            return jsonify({
                "success": False,
                "error": f"Input preparation failed: {str(e)}"
            }), 500
        
        # Generate speech with timeout and error handling
        try:
            logger.info("Starting speech generation...")
            
            with torch.no_grad():
                # Set generation parameters for stability
                generation_config = {
                    "do_sample": True,
                    "max_length": 1024,
                    "pad_token_id": processor.tokenizer.eos_token_id,
                    "eos_token_id": processor.tokenizer.eos_token_id,
                }
                
                # Generate audio
                audio_array = model.generate(
                    **inputs,
                    **generation_config
                )
            
            logger.info("Speech generation completed")
            
        except Exception as e:
            logger.error(f"Speech generation failed: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return jsonify({
                "success": False,
                "error": f"Speech generation failed: {str(e)}"
            }), 500
        
        # Process audio output
        try:
            # Convert to numpy and ensure correct format
            if torch.is_tensor(audio_array):
                audio_array = audio_array.cpu().numpy()
            
            # Handle different output shapes
            if audio_array.ndim > 1:
                audio_array = audio_array.squeeze()
            
            # Ensure we have valid audio data
            if audio_array.size == 0:
                raise ValueError("Generated audio is empty")
            
            # Normalize audio to prevent clipping
            if np.max(np.abs(audio_array)) > 0:
                audio_array = audio_array / np.max(np.abs(audio_array)) * 0.8
            else:
                raise ValueError("Generated audio contains only silence")
            
            # Convert to 16-bit PCM
            audio_int16 = (audio_array * 32767).astype(np.int16)
            
            logger.info(f"Audio processed: shape={audio_int16.shape}, duration={len(audio_int16)/24000:.2f}s")
            
        except Exception as e:
            logger.error(f"Audio processing failed: {str(e)}")
            return jsonify({
                "success": False,
                "error": f"Audio processing failed: {str(e)}"
            }), 500
        
        # Create WAV file and encode as base64
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                # Write WAV file
                wavfile.write(tmp_file.name, 24000, audio_int16)  # Bark uses 24kHz
                
                # Read file and encode as base64
                with open(tmp_file.name, 'rb') as f:
                    audio_data = f.read()
                    audio_base64 = base64.b64encode(audio_data).decode('utf-8')
                
                # Clean up temp file
                os.unlink(tmp_file.name)
            
            # Calculate duration
            duration = len(audio_int16) / 24000  # 24kHz sample rate
            
            logger.info(f"Audio file created successfully: {len(audio_data)} bytes, {duration:.2f}s")
            
        except Exception as e:
            logger.error(f"Audio file creation failed: {str(e)}")
            return jsonify({
                "success": False,
                "error": f"Audio file creation failed: {str(e)}"
            }), 500
        
        # Return successful response
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
                "file_size": f"{len(audio_data) / 1024:.1f} KB",
                "device": str(device)
            },
            "timestamp": datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Unexpected error in generate_speech: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({
            "success": False,
            "error": f"Unexpected error: {str(e)}"
        }), 500

@app.route('/generate-music', methods=['POST'])
def generate_music():
    """Generate music from text prompt with enhanced procedural generation"""
    try:
        data = request.get_json()
        
        if not data or 'prompt' not in data:
            return jsonify({
                "success": False,
                "error": "Missing 'prompt' parameter"
            }), 400
        
        prompt = data['prompt'].strip()
        duration = min(data.get('duration', 10), 30)  # Cap at 30 seconds
        
        if not prompt:
            return jsonify({
                "success": False,
                "error": "Prompt cannot be empty"
            }), 400
        
        logger.info(f"Generating music for prompt: {prompt}")
        
        # Generate enhanced procedural music
        sample_rate = 44100
        samples = int(sample_rate * duration)
        
        # Create time array
        t = np.linspace(0, duration, samples)
        
        # Analyze prompt for music style and generate accordingly
        prompt_lower = prompt.lower()
        
        # Determine musical parameters based on prompt
        if 'electronic' in prompt_lower or 'dance' in prompt_lower or 'edm' in prompt_lower:
            # Electronic/Dance music
            base_freq = 220  # A3
            chord_freqs = [220, 277, 330, 440]  # A major chord
            bass_freq = 55  # A1
            beat_freq = 4.0  # Fast beat
            style = "Electronic Dance"
        elif 'classical' in prompt_lower or 'piano' in prompt_lower or 'orchestral' in prompt_lower:
            # Classical music
            base_freq = 261  # C4
            chord_freqs = [261, 329, 392, 523]  # C major chord
            bass_freq = 65  # C2
            beat_freq = 1.5  # Moderate beat
            style = "Classical"
        elif 'rock' in prompt_lower or 'guitar' in prompt_lower or 'metal' in prompt_lower:
            # Rock music
            base_freq = 329  # E4
            chord_freqs = [329, 415, 493, 659]  # E major chord
            bass_freq = 82  # E2
            beat_freq = 2.5  # Rock beat
            style = "Rock"
        elif 'jazz' in prompt_lower or 'blues' in prompt_lower:
            # Jazz music
            base_freq = 293  # D4
            chord_freqs = [293, 369, 440, 554]  # D7 chord
            bass_freq = 73  # D2
            beat_freq = 2.0  # Swing beat
            style = "Jazz"
        elif 'ambient' in prompt_lower or 'calm' in prompt_lower or 'relaxing' in prompt_lower:
            # Ambient music
            base_freq = 440  # A4
            chord_freqs = [440, 523, 659, 880]  # A major chord
            bass_freq = 110  # A2
            beat_freq = 0.5  # Very slow
            style = "Ambient"
        else:
            # Default pleasant melody
            base_freq = 440  # A4
            chord_freqs = [440, 554, 659, 880]  # A major chord
            bass_freq = 110  # A2
            beat_freq = 2.0  # Medium beat
            style = "Melodic"
        
        # Initialize audio array
        audio = np.zeros(samples)
        
        # Add main melody with harmonics
        for i, freq in enumerate(chord_freqs):
            amplitude = 0.15 / (i + 1)  # Decreasing amplitude for harmonics
            phase = np.random.random() * 2 * np.pi  # Random phase for richness
            
            # Main tone
            audio += amplitude * np.sin(2 * np.pi * freq * t + phase)
            
            # Add subtle harmonics
            audio += amplitude * 0.3 * np.sin(2 * np.pi * freq * 2 * t + phase)
            audio += amplitude * 0.1 * np.sin(2 * np.pi * freq * 3 * t + phase)
        
        # Add bass line with rhythm
        bass_rhythm = np.sin(2 * np.pi * beat_freq * t)
        bass_envelope = 0.5 + 0.5 * np.maximum(0, bass_rhythm)
        audio += 0.2 * np.sin(2 * np.pi * bass_freq * t) * bass_envelope
        
        # Add rhythmic elements
        if 'upbeat' in prompt_lower or 'fast' in prompt_lower:
            beat_freq *= 1.5
        elif 'slow' in prompt_lower or 'calm' in prompt_lower:
            beat_freq *= 0.7
            
        # Create beat envelope
        beat_envelope = 0.7 + 0.3 * np.sin(2 * np.pi * beat_freq * t)
        audio *= beat_envelope
        
        # Add texture based on style
        if style == "Electronic Dance":
            # Add electronic-style effects
            noise = np.random.normal(0, 0.02, samples)
            audio += noise * np.sin(2 * np.pi * 8 * t)  # High-freq texture
        elif style == "Rock":
            # Add distortion-like effect
            audio = np.tanh(audio * 2) * 0.8
        elif style == "Ambient":
            # Add reverb-like effect
            reverb = np.convolve(audio, np.exp(-np.linspace(0, 2, 1000)), mode='same')
            audio = 0.7 * audio + 0.3 * reverb[:len(audio)]
        
        # Apply fade in/out for smooth start and end
        fade_samples = int(0.1 * sample_rate)  # 0.1 second fade
        fade_in = np.linspace(0, 1, fade_samples)
        fade_out = np.linspace(1, 0, fade_samples)
        
        audio[:fade_samples] *= fade_in
        audio[-fade_samples:] *= fade_out
        
        # Normalize to prevent clipping
        if np.max(np.abs(audio)) > 0:
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
                "style": style,
                "duration": duration,
                "sample_rate": sample_rate,
                "format": "wav",
                "file_size": f"{len(audio_data) / 1024:.1f} KB",
                "base_frequency": f"{base_freq} Hz",
                "beat_frequency": f"{beat_freq} Hz"
            },
            "timestamp": datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Music generation error: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
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
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({
        "success": False,
        "error": "Internal server error"
    }), 500

@app.errorhandler(Exception)
def handle_exception(e):
    logger.error(f"Unhandled exception: {str(e)}")
    logger.error(f"Traceback: {traceback.format_exc()}")
    return jsonify({
        "success": False,
        "error": f"Unexpected error: {str(e)}"
    }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"Starting Mythiq Audio Creator v2.0 on port {port}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    app.run(host='0.0.0.0', port=port, debug=False)
