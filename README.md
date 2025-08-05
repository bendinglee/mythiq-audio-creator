Mythiq Audio Creator

AI-powered audio generation service for the Mythiq platform, featuring REAL speech synthesis and music generation capabilities.

🎯 WORKING FEATURES

✅ Speech Generation: High-quality text-to-speech using Bark AI


✅ Music Generation: Procedural music generation based on prompt analysis


✅ Multiple Voice Presets: 3 different voice options for speech generation


✅ Real Audio Files: Returns actual playable WAV files via base64 encoding


✅ Browser Compatible: Audio plays immediately in web browsers

🤖 Models Used

• Bark (Suno AI): Advanced text-to-speech with emotional expression


• Procedural Music Generator: Smart music generation based on prompt keywords

🚀 API Endpoints

Health Check

Plain Text


GET /health


Returns service status and model availability.

Response:

JSON


{
  "status": "online",
  "service": "mythiq-audio-creator",
  "models_loaded": {"bark": true},
  "device": "cpu",
  "cuda_available": false,
  "message": "Audio generation service ready"
}


Generate Speech

Plain Text


POST /generate-speech
Content-Type: application/json

{
  "text": "Hello, this is a test of speech generation!",
  "voice_preset": "v2/en_speaker_1"
}


Response:

JSON


{
  "success": true,
  "message": "🗣️ Speech generated successfully!",
  "audio_data": "data:audio/wav;base64,UklGRnoGAAB...",
  "generation_info": {
    "text": "Hello, this is a test...",
    "voice_preset": "v2/en_speaker_1",
    "duration": 3.2,
    "sample_rate": 24000,
    "format": "wav",
    "file_size": "76.8 KB"
  }
}


Generate Music

Plain Text


POST /generate-music
Content-Type: application/json

{
  "prompt": "upbeat electronic dance music",
  "duration": 10
}


Response:

JSON


{
  "success": true,
  "message": "🎶 Music generated successfully!",
  "audio_data": "data:audio/wav;base64,UklGRnoGAAB...",
  "generation_info": {
    "prompt": "upbeat electronic dance music",
    "duration": 10,
    "sample_rate": 44100,
    "format": "wav",
    "file_size": "441.0 KB"
  }
}


Get Voice Presets

Plain Text


GET /voice-presets


Response:

JSON


{
  "success": true,
  "presets": [
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
}


🔧 Environment Variables

No additional environment variables required. The service uses default model configurations.

🚀 Deployment

Railway Deployment

1.
Create a new Railway project

2.
Connect your GitHub repository

3.
Railway will automatically detect the configuration from railway.json

4.
Deploy and wait for the service to start (initial deployment may take 10-15 minutes due to model downloads)

Local Development

Install dependencies:

Bash


pip install -r requirements.txt


Run the application:

Bash


python app.py


The service will be available at http://localhost:5000

🤖 Model Information

Bark Model

• Size: ~2GB


• Sample Rate: 24kHz


• Languages: English, Chinese, French, German, Spanish, and more


• Features: Emotional expression, sound effects, music

Procedural Music Generator

• Sample Rate: 44.1kHz


• Duration: Up to 30 seconds per generation


• Styles: Analyzes prompts for electronic, classical, rock, and ambient styles


• Features: Dynamic chord progressions, rhythm patterns, and harmonics

⚡ Performance Notes

• First Request: May take 30-60 seconds due to model loading


• Subsequent Requests: 5-15 seconds depending on content length


• GPU Acceleration: Automatically uses GPU if available


• Memory Requirements: 4-8GB RAM recommended for optimal performance

🛠️ Error Handling

The service includes comprehensive error handling and logging. Check the logs for detailed error information if requests fail.

🌐 CORS Configuration

The service is configured to accept requests from:
• https://mythiq-ui-production.up.railway.app
• http://localhost:5173
• http://localhost:3000

🎉 What's New

✅ REAL Audio Generation: No more fake responses - actual WAV files are generated


✅ Base64 Encoding: Audio files are returned as data URLs for immediate browser playback


✅ Smart Music Generation: Procedural music that adapts to prompt keywords


✅ Professional Error Handling: Comprehensive logging and error responses


✅ Production Ready: Optimized for Railway deployment with proper resource management

📄 License

This service is part of the Mythiq AI platform.

