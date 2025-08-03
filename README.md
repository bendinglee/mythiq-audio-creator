Mythiq Audio Creator

AI-powered audio generation service for the Mythiq platform, featuring speech synthesis and music generation capabilities.

Features

•
Speech Generation: High-quality text-to-speech using Bark AI

•
Music Generation: AI-generated music using MusicGen

•
Multiple Voice Presets: Various voice options for speech generation

•
Multilingual Support: Support for multiple languages

•
RESTful API: Easy integration with frontend applications

Models Used

•
Bark (Suno AI): Advanced text-to-speech with emotional expression

•
MusicGen (Meta): High-quality music generation from text prompts

API Endpoints

Health Check

Plain Text


GET /health


Returns service status and model availability.

Generate Speech

Plain Text


POST /generate-speech
Content-Type: application/json

{
  "text": "Hello, this is a test of speech generation!",
  "voice_preset": "v2/en_speaker_6"
}


Generate Music

Plain Text


POST /generate-music
Content-Type: application/json

{
  "prompt": "upbeat electronic dance music",
  "duration": 10
}


Get Voice Presets

Plain Text


GET /voice-presets


Returns available voice presets for speech generation.

Environment Variables

No additional environment variables required. The service uses default model configurations.

Deployment

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

1.
Install dependencies:

Bash


pip install -r requirements.txt


1.
Run the application:

Bash


python app.py


The service will be available at http://localhost:5000

Model Information

Bark Model

•
Size: ~2GB

•
Sample Rate: 24kHz

•
Languages: English, Chinese, French, German, Spanish, and more

•
Features: Emotional expression, sound effects, music

MusicGen Model

•
Size: ~1.5GB (small model)

•
Sample Rate: 32kHz

•
Duration: Up to 30 seconds per generation

•
Styles: Various music genres and styles

Performance Notes

•
First Request: May take 30-60 seconds due to model loading

•
Subsequent Requests: 5-15 seconds depending on content length

•
GPU Acceleration: Automatically uses GPU if available

•
Memory Requirements: 4-8GB RAM recommended

Error Handling

The service includes comprehensive error handling and logging. Check the logs for detailed error information if requests fail.

CORS Configuration

The service is configured to accept requests from:

•
https://mythiq-ui-production.up.railway.app

•
http://localhost:5173

•
http://localhost:3000

License

This service is part of the Mythiq AI platform.

