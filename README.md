# AI Trip Planner

A sophisticated trip planning application powered by CrewAI agents with Arize tracing. This application uses intelligent agent routing to provide specialized travel assistance including destination research, itinerary planning, budget advice, and local recommendations.

## Features

- 🤖 **AI Agent Routing**: Automatically routes requests to specialized travel agents
- 🔍 **Destination Research**: Weather, attractions, culture, and practical information
- 📅 **Itinerary Planning**: Detailed day-by-day travel plans
- 💰 **Budget Advisory**: Cost breakdowns and money-saving tips
- 🍽️ **Local Experiences**: Authentic restaurants and cultural activities
- 📊 **Arize Tracing**: Complete observability of agent interactions with Arize
- 🎨 **Modern UI**: Beautiful Material-UI interface

## Quick Start

### Prerequisites

- Python 3.8+ with conda/pip
- Node.js 16+
- OpenAI API Key
- Arize Space ID and API Key (get from [app.arize.com](https://app.arize.com))

### 1. Backend Setup (Conda Environment)

```bash
cd trip_planner/backend

# Install dependencies in your conda environment
pip install -r requirements.txt

# Create .env file with your API keys
OPENAI_API_KEY=your_openai_api_key_here
SERPER_API_KEY=your_serper_api_key_here  # Optional for web search
ARIZE_SPACE_ID=your_arize_space_id_here
ARIZE_API_KEY=your_arize_api_key_here

# Start the backend
python main.py
```

### 2. Frontend Setup

```bash
cd trip_planner/frontend
npm install
npm start
```

Visit `http://localhost:3000` to use the application!

## API Usage

```bash
curl -X POST "http://localhost:8000/plan-trip" \
  -H "Content-Type: application/json" \
  -d '{
    "destination": "Tokyo, Japan",
    "duration": "7 days",
    "interests": "food, culture"
  }'
```

## View Traces

Check your agent interactions at: [app.arize.com](https://app.arize.com)

## Getting Arize Credentials

1. Sign up at [app.arize.com](https://app.arize.com)
2. Create a new space or use an existing one
3. Go to Space Settings to find your:
   - **Space ID**: Found in space settings
   - **API Key**: Generate one in space settings

## Agent Types

The application automatically routes requests to specialized agents:

- **🔍 Research Agent**: "What's the weather in Iceland?"
- **📅 Itinerary Agent**: "Create a 5-day Rome itinerary"
- **💰 Budget Agent**: "Budget for Thailand trip"
- **🍽️ Local Agent**: "Best restaurants in Tokyo"

## Docker Deployment

```bash
cd trip_planner
docker-compose up
```

## Testing

```bash
cd trip_planner
python test_api.py
```
