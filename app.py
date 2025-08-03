from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configure CORS
CORS(app, origins=[
    'https://mythiq-ui-production.up.railway.app',
    'http://localhost:5173',
    'http://localhost:3000'
] )

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        return jsonify({
            "status": "online",
            "service": "mythiq-audio-creator",
            "timestamp": datetime.now().isoformat(),
            "message": "Service is running - AI models will be loaded on first request",
            "features": [
                "Basic service running",
                "Ready for AI model integration"
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
    """Generate speech - placeholder for now"""
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({
                "success": False,
                "error": "Missing 'text' parameter"
            }), 400
        
        text = data['text']
        
        # Placeholder response
        return jsonify({
            "success": True,
            "message": f"Speech generation requested for: {text[:50]}...",
            "status": "AI models will be integrated in next update",
            "text": text
        })
        
    except Exception as e:
        logger.error(f"Speech generation error: {str(e)}")
        return jsonify({
            "success": False,
            "error": f"Speech generation failed: {str(e)}"
        }), 500

@app.route('/generate-music', methods=['POST'])
def generate_music():
    """Generate music - placeholder for now"""
    try:
        data = request.get_json()
        
        if not data or 'prompt' not in data:
            return jsonify({
                "success": False,
                "error": "Missing 'prompt' parameter"
            }), 400
        
        prompt = data['prompt']
        
        # Placeholder response
        return jsonify({
            "success": True,
            "message": f"Music generation requested for: {prompt}",
            "status": "AI models will be integrated in next update",
            "prompt": prompt
        })
        
    except Exception as e:
        logger.error(f"Music generation error: {str(e)}")
        return jsonify({
            "success": False,
            "error": f"Music generation failed: {str(e)}"
        }), 500

@app.route('/voice-presets', methods=['GET'])
def get_voice_presets():
    """Get available voice presets"""
    presets = {
        "english": [
            {"id": "v2/en_speaker_0", "name": "English Speaker 0", "gender": "male"},
            {"id": "v2/en_speaker_1", "name": "English Speaker 1", "gender": "female"},
            {"id": "v2/en_speaker_6", "name": "English Speaker 6", "gender": "male"},
        ]
    }
    
    return jsonify({
        "success": True,
        "presets": presets,
        "status": "Voice presets available - AI integration coming next"
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
    logger.info(f"Starting Mythiq Audio Creator on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)
