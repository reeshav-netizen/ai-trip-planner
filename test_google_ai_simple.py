#!/usr/bin/env python3
"""
Simple test to verify Google AI Studio is working
"""

import os
from dotenv import load_dotenv
import google.generativeai as genai

def test_google_ai():
    """Test Google AI Studio with a simple completion"""
    print("🔍 Testing Google AI Studio (Gemini)...")
    
    # Load environment variables
    load_dotenv("backend/.env")
    
    # Get API key
    api_key = os.getenv("GOOGLE_API_KEY")
    
    if not api_key:
        print("❌ GOOGLE_API_KEY not found")
        return False
    
    try:
        # Configure Google AI
        genai.configure(api_key=api_key)
        
        # Test with a simple completion
        print("🧪 Testing chat completion...")
        
        # Create a model instance
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Generate content
        response = model.generate_content(
            "Hello! Please respond with 'Google AI Studio is working perfectly!' if you can see this message."
        )
        
        result = response.text
        print(f"✅ Google AI Studio is working!")
        print(f"🤖 Response: {result}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """Run the test"""
    print("🚀 Google AI Studio Simple Test\n")
    
    success = test_google_ai()
    
    print("\n" + "="*50)
    if success:
        print("🎉 SUCCESS: Google AI Studio is working!")
        print("✅ Your API key is valid and working")
        print("✅ The system will use Google AI Studio as default")
    else:
        print("❌ FAILED: Google AI Studio test failed")
    print("="*50)

if __name__ == "__main__":
    main() 