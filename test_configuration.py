#!/usr/bin/env python3
"""
Test script to check the current configuration and show what needs to be set up
"""

import os
import sys
from dotenv import load_dotenv

def check_python_version():
    """Check if Python version is compatible"""
    print("🐍 Python Version Check")
    print(f"📝 Current version: {sys.version}")
    
    if sys.version_info >= (3, 9):
        print("✅ Python version is compatible with Google AI Studio")
        return True
    else:
        print("❌ Python version too old for Google AI Studio")
        print("📝 Google AI Studio requires Python 3.9 or higher")
        print("📝 See GOOGLE_AI_SETUP.md for upgrade instructions")
        return False

def check_environment():
    """Check environment configuration"""
    print("\n🔧 Environment Configuration")
    
    # Load environment variables
    load_dotenv("backend/.env")
    
    # Check API keys
    google_key = os.getenv("GOOGLE_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    
    print("📋 API Keys Status:")
    
    if google_key and google_key != "your_google_ai_studio_api_key_here":
        print("✅ GOOGLE_API_KEY: Configured")
    else:
        print("❌ GOOGLE_API_KEY: Not configured")
        print("   📝 Get key from: https://aistudio.google.com/")
    
    if openai_key and openai_key != "your_openai_api_key_here":
        print("✅ OPENAI_API_KEY: Configured (fallback)")
    else:
        print("⚠️ OPENAI_API_KEY: Not configured (optional fallback)")
    
    return google_key and google_key != "your_google_ai_studio_api_key_here"

def check_packages():
    """Check if required packages are installed"""
    print("\n📦 Package Dependencies")
    
    packages = {
        "google-generativeai": "Google AI Studio support",
        "langchain-google-genai": "LangChain Google AI integration",
        "openai": "OpenAI fallback support"
    }
    
    all_installed = True
    
    for package, description in packages.items():
        try:
            if package == "google-generativeai":
                import google.generativeai
                print(f"✅ {package}: Installed ({description})")
            elif package == "langchain-google-genai":
                import langchain_google_genai
                print(f"✅ {package}: Installed ({description})")
            elif package == "openai":
                import openai
                print(f"✅ {package}: Installed ({description})")
        except ImportError:
            print(f"❌ {package}: Not installed ({description})")
            all_installed = False
    
    return all_installed

def main():
    """Run configuration check"""
    print("🚀 AI Trip Planner - Configuration Check\n")
    
    # Check Python version
    python_ok = check_python_version()
    
    # Check environment
    env_ok = check_environment()
    
    # Check packages
    packages_ok = check_packages()
    
    print("\n" + "="*50)
    print("📊 Configuration Summary:")
    
    if python_ok and env_ok and packages_ok:
        print("🎉 All systems ready!")
        print("✅ You can now use Google AI Studio as the default")
        print("🚀 Run: python test_google_ai_access.py")
    else:
        print("⚠️ Some configuration needed:")
        if not python_ok:
            print("   - Update Python to 3.9+")
        if not env_ok:
            print("   - Add Google AI Studio API key")
        if not packages_ok:
            print("   - Install missing packages")
        print("\n📝 See GOOGLE_AI_SETUP.md for detailed instructions")
    
    print("="*50)

if __name__ == "__main__":
    main() 