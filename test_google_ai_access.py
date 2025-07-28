#!/usr/bin/env python3
"""
Test script to verify Google AI Studio (Gemini) API access
"""

import os
import sys
from dotenv import load_dotenv

# Check Python version first
if sys.version_info < (3, 9):
    print("❌ Google AI Studio requires Python 3.9 or higher")
    print(f"📝 Current version: {sys.version}")
    print("📝 Please update Python and try again")
    print("📝 See GOOGLE_AI_SETUP.md for instructions")
    sys.exit(1)

try:
    import google.generativeai as genai
except ImportError:
    print("❌ google-generativeai package not installed")
    print("📝 Install with: pip install google-generativeai")
    print("📝 Note: Requires Python 3.9+")
    sys.exit(1)

def test_google_ai_access():
    """Test Google AI Studio access with a simple completion"""
    print("🔍 Testing Google AI Studio (Gemini) API access...")
    
    # Load environment variables from the correct location
    load_dotenv("backend/.env")
    
    # Get API key
    api_key = os.getenv("GOOGLE_API_KEY")
    
    if not api_key or api_key == "your_google_ai_studio_api_key_here":
        print("❌ GOOGLE_API_KEY not found or not configured")
        print("📝 Please set your Google AI Studio API key in the environment or .env file")
        print("📝 You can copy backend/env_example.txt to backend/.env and update it")
        return False
    
    try:
        # Configure Google AI
        genai.configure(api_key=api_key)
        
        # Test with a simple completion
        print("🧪 Testing with a simple chat completion...")
        
        # Create a model instance
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Generate content
        response = model.generate_content(
            "Hello! Please respond with 'Google AI Studio access is working!' if you can see this message."
        )
        
        result = response.text
        print(f"✅ Google AI Studio API is working!")
        print(f"🤖 Response: {result}")
        
        # Test model list access
        print("\n📋 Testing model list access...")
        models = genai.list_models()
        print(f"✅ Successfully retrieved {len(models)} available models")
        
        # Show some available models
        model_names = [model.name for model in models[:5]]
        print(f"📝 Sample models: {', '.join(model_names)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Google AI Studio API error: {e}")
        return False

def test_environment_setup():
    """Check if environment is properly configured"""
    print("🔧 Checking environment setup...")
    
    # Check if .env file exists
    env_file = "backend/.env"
    if os.path.exists(env_file):
        print(f"✅ Found {env_file}")
    else:
        print(f"⚠️ {env_file} not found")
        print("📝 You can create it by copying backend/env_example.txt")
    
    # Check required environment variables
    load_dotenv("backend/.env")
    required_vars = ["GOOGLE_API_KEY"]
    optional_vars = ["OPENAI_API_KEY", "GROQ_API_KEY", "TAVILY_API_KEY", "ARIZE_SPACE_ID", "ARIZE_API_KEY"]
    
    print("\n📋 Environment variables:")
    for var in required_vars:
        value = os.getenv(var)
        if value and value != f"your_{var.lower()}_here":
            print(f"✅ {var}: Set")
        else:
            print(f"❌ {var}: Not set or using placeholder")
    
    for var in optional_vars:
        value = os.getenv(var)
        if value and value != f"your_{var.lower()}_here":
            print(f"✅ {var}: Set")
        else:
            print(f"⚠️ {var}: Not set (optional)")

def main():
    """Run the Google AI Studio access test"""
    print("🚀 Google AI Studio (Gemini) Access Test\n")
    
    # Check environment setup
    test_environment_setup()
    print()
    
    # Test Google AI Studio access
    success = test_google_ai_access()
    
    print("\n" + "="*50)
    if success:
        print("🎉 Google AI Studio access test PASSED!")
        print("✅ Your Google AI Studio API key is working correctly")
    else:
        print("❌ Google AI Studio access test FAILED!")
        print("📝 Please check your API key and try again")
    
    print("="*50)

if __name__ == "__main__":
    main() 