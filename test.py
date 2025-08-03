#!/usr/bin/env python3
"""
Test script for Mythiq Audio Creator service
Run this to test the service locally before deployment
"""

import requests
import json
import time
import base64
import os

# Configuration
BASE_URL = "http://localhost:5000"  # Change to your deployed URL for production testing
TEST_AUDIO_DIR = "test_audio_output"

def create_output_dir():
    """Create directory for test audio output"""
    if not os.path.exists(TEST_AUDIO_DIR):
        os.makedirs(TEST_AUDIO_DIR)

def save_audio_from_base64(audio_data, filename):
    """Save base64 audio data to file"""
    try:
        # Remove the data URL prefix
        if audio_data.startswith('data:audio/wav;base64,'):
            audio_data = audio_data[len('data:audio/wav;base64,'):]
        
        # Decode base64
        audio_bytes = base64.b64decode(audio_data)
        
        # Save to file
        filepath = os.path.join(TEST_AUDIO_DIR, filename)
        with open(filepath, 'wb') as f:
            f.write(audio_bytes)
        
        print(f"✅ Audio saved to: {filepath}")
        return True
    except Exception as e:
        print(f"❌ Error saving audio: {e}")
        return False

def test_health_check():
    """Test the health check endpoint"""
    print("\n🔍 Testing Health Check...")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Health check passed!")
            print(f"   Status: {data.get('status')}")
            print(f"   Service: {data.get('service')}")
            print(f"   Models: {data.get('models')}")
            print(f"   GPU Available: {data.get('gpu_available')}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Health check error: {e}")
        return False

def test_speech_generation():
    """Test speech generation endpoint"""
    print("\n🎤 Testing Speech Generation...")
    
    test_cases = [
        {
            "text": "Hello! This is a test of the Mythiq audio generation system.",
            "voice_preset": "v2/en_speaker_6",
            "filename": "test_speech_1.wav"
        },
        {
            "text": "Welcome to Mythiq AI! We can generate realistic speech with emotions.",
            "voice_preset": "v2/en_speaker_1",
            "filename": "test_speech_2.wav"
        },
        {
            "text": "This is amazing! [laughs] The AI can even generate laughter and emotions!",
            "voice_preset": "v2/en_speaker_3",
            "filename": "test_speech_emotions.wav"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n   Test {i}: {test_case['text'][:50]}...")
        
        try:
            start_time = time.time()
            response = requests.post(
                f"{BASE_URL}/generate-speech",
                json=test_case,
                timeout=120
            )
            end_time = time.time()
            
            if response.status_code == 200:
                data = response.json()
                if data.get('success'):
                    print(f"   ✅ Speech generated in {end_time - start_time:.1f}s")
                    print(f"      Duration: {data.get('duration', 0):.1f}s")
                    print(f"      Sample Rate: {data.get('sample_rate')}Hz")
                    
                    # Save audio file
                    if save_audio_from_base64(data.get('audio_data'), test_case['filename']):
                        print(f"      Audio saved as: {test_case['filename']}")
                else:
                    print(f"   ❌ Speech generation failed: {data.get('error')}")
            else:
                print(f"   ❌ Request failed: {response.status_code}")
                print(f"      Response: {response.text}")
                
        except requests.exceptions.RequestException as e:
            print(f"   ❌ Request error: {e}")

def test_music_generation():
    """Test music generation endpoint"""
    print("\n🎵 Testing Music Generation...")
    
    test_cases = [
        {
            "prompt": "upbeat electronic dance music",
            "duration": 10,
            "filename": "test_music_edm.wav"
        },
        {
            "prompt": "peaceful acoustic guitar melody",
            "duration": 8,
            "filename": "test_music_acoustic.wav"
        },
        {
            "prompt": "energetic rock music with drums",
            "duration": 12,
            "filename": "test_music_rock.wav"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n   Test {i}: {test_case['prompt']}")
        
        try:
            start_time = time.time()
            response = requests.post(
                f"{BASE_URL}/generate-music",
                json=test_case,
                timeout=180
            )
            end_time = time.time()
            
            if response.status_code == 200:
                data = response.json()
                if data.get('success'):
                    print(f"   ✅ Music generated in {end_time - start_time:.1f}s")
                    print(f"      Duration: {data.get('duration')}s")
                    print(f"      Sample Rate: {data.get('sample_rate')}Hz")
                    
                    # Save audio file
                    if save_audio_from_base64(data.get('audio_data'), test_case['filename']):
                        print(f"      Music saved as: {test_case['filename']}")
                else:
                    print(f"   ❌ Music generation failed: {data.get('error')}")
            else:
                print(f"   ❌ Request failed: {response.status_code}")
                print(f"      Response: {response.text}")
                
        except requests.exceptions.RequestException as e:
            print(f"   ❌ Request error: {e}")

def test_voice_presets():
    """Test voice presets endpoint"""
    print("\n🎭 Testing Voice Presets...")
    
    try:
        response = requests.get(f"{BASE_URL}/voice-presets", timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                presets = data.get('presets', {})
                print("✅ Voice presets retrieved!")
                print(f"   English voices: {len(presets.get('english', []))}")
                print(f"   Multilingual voices: {len(presets.get('multilingual', []))}")
                
                # Show a few examples
                english_voices = presets.get('english', [])[:3]
                for voice in english_voices:
                    print(f"      - {voice['name']} ({voice['id']})")
            else:
                print(f"❌ Failed to get voice presets: {data.get('error')}")
        else:
            print(f"❌ Request failed: {response.status_code}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request error: {e}")

def main():
    """Run all tests"""
    print("🚀 Starting Mythiq Audio Creator Tests")
    print(f"   Testing service at: {BASE_URL}")
    
    # Create output directory
    create_output_dir()
    
    # Run tests
    health_ok = test_health_check()
    
    if health_ok:
        test_voice_presets()
        test_speech_generation()
        test_music_generation()
        
        print("\n🎉 All tests completed!")
        print(f"   Check the '{TEST_AUDIO_DIR}' directory for generated audio files.")
    else:
        print("\n❌ Health check failed. Service may not be running or ready.")
        print("   Make sure the service is started and models are loaded.")

if __name__ == "__main__":
    main()
