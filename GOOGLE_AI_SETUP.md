# Google AI Studio (Gemini) Setup Guide

## 🚨 Important: Python Version Requirement

**Google AI Studio requires Python 3.9 or higher.** You're currently using Python 3.7.4.

## 🔧 Setup Options

### Option 1: Update Python Version (Recommended)

1. **Install Python 3.9+**
   ```bash
   # Using Homebrew (macOS)
   brew install python@3.11
   
   # Or download from python.org
   # https://www.python.org/downloads/
   ```

2. **Create a new virtual environment**
   ```bash
   # Create new venv with Python 3.11
   python3.11 -m venv venv_google_ai
   source venv_google_ai/bin/activate
   
   # Install requirements
   pip install -r backend/requirements.txt
   ```

### Option 2: Use Docker (Alternative)

The project already has Docker support. You can run it with:
```bash
docker-compose up
```

## 🔑 Getting Google AI Studio API Key

1. **Visit Google AI Studio**
   - Go to [Google AI Studio](https://aistudio.google.com/)
   - Sign in with your Google account

2. **Get API Key**
   - Click on "Get API key" in the top right
   - Create a new API key or use existing one
   - Copy the API key

3. **Update Environment**
   ```bash
   # Edit backend/.env
   GOOGLE_API_KEY=your_actual_api_key_here
   ```

## 🧪 Testing Google AI Studio

After setup, test with:
```bash
python test_google_ai_access.py
```

## 🔄 Fallback Configuration

The system is configured to:
1. **Try Google AI Studio first** (preferred)
2. **Fallback to OpenAI** if Google AI Studio fails
3. **Show clear error** if no valid API keys

## 📋 Current Status

- ✅ Backend code updated for Google AI Studio
- ✅ Environment configuration ready
- ❌ Python version needs update (3.7 → 3.9+)
- ⏳ Ready to test once Python is updated

## 🚀 Quick Start (After Python Update)

1. **Update Python to 3.9+**
2. **Create new virtual environment**
3. **Install requirements**: `pip install -r backend/requirements.txt`
4. **Add Google AI API key** to `backend/.env`
5. **Test**: `python test_google_ai_access.py`
6. **Start backend**: `cd backend && python main.py` 