from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import os
import asyncio
import uvicorn
from contextlib import asynccontextmanager
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Arize and tracing imports
from arize.otel import register
from openinference.instrumentation.crewai import CrewAIInstrumentor
from openinference.instrumentation.langchain import LangChainInstrumentor

# CrewAI imports
from crewai import Agent, Crew, Process, Task
from crewai.flow import Flow, listen, router, start
from crewai_tools import SerperDevTool


# Initialize Arize tracing
def setup_tracing():
    tracer_provider = register(
        space_id=os.getenv("ARIZE_SPACE_ID", "your-space-id"),  # Set in your .env file
        api_key=os.getenv("ARIZE_API_KEY", "your-arize-api-key"),  # Set in your .env file
        project_name="trip-planner"
    )

    # Instrument CrewAI and LangChain for deeper visibility
    CrewAIInstrumentor().instrument(tracer_provider=tracer_provider)
    LangChainInstrumentor().instrument(tracer_provider=tracer_provider)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Setup tracing on startup
    setup_tracing()
    yield


app = FastAPI(title="Trip Planner API", lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models
class TripRequest(BaseModel):
    destination: str
    duration: str
    budget: Optional[str] = None
    interests: Optional[str] = None
    travel_style: Optional[str] = None


class TripResponse(BaseModel):
    agent_type: str
    result: str
    route_taken: str


# Define specialized travel agents
destination_researcher = Agent(
    role="Destination Research Specialist",
    goal="Research destinations, attractions, weather, and local culture information.",
    backstory="Expert travel researcher with deep knowledge of global destinations, local customs, weather patterns, and hidden gems.",
    verbose=True,
    tools=[SerperDevTool()] if os.getenv("SERPER_API_KEY") else []
)

itinerary_planner = Agent(
    role="Travel Itinerary Planner",
    goal="Create detailed day-by-day travel itineraries based on user preferences and research.",
    backstory="Experienced travel planner who creates perfectly balanced itineraries considering time, budget, and traveler interests.",
    verbose=True
)

budget_advisor = Agent(
    role="Travel Budget Advisor",
    goal="Provide detailed budget breakdowns and money-saving tips for travel plans.",
    backstory="Financial travel expert who helps travelers maximize their experiences while staying within budget.",
    verbose=True
)

local_expert = Agent(
    role="Local Experience Curator",
    goal="Recommend authentic local experiences, restaurants, and cultural activities.",
    backstory="Cultural ambassador with insider knowledge of local experiences, authentic restaurants, and off-the-beaten-path activities.",
    verbose=True
)


# Router agent for trip planning
router_agent = Agent(
    role="Trip Planning Router",
    goal="Classify trip planning requests and route them to the appropriate specialist.",
    backstory="Intelligent routing system that understands different types of travel planning needs.",
    verbose=False
)


def route_trip_request(user_input: str, router):
    """Route the trip request to appropriate agent"""
    router_task = Task(
        description=f"Classify this trip planning request: '{user_input}'. Determine if this is primarily about: 'research' (destination info/weather/attractions), 'itinerary' (day-by-day planning), 'budget' (cost planning/money advice), or 'local' (authentic experiences/restaurants).",
        agent=router,
        expected_output="One word: 'research', 'itinerary', 'budget', or 'local'"
    )

    router_crew = Crew(
        agents=[router],
        tasks=[router_task],
        process=Process.sequential,
        verbose=False
    )

    router_results = router_crew.kickoff()
    return router_results


def determine_task_type(router_results):
    """Extract the task type from router results"""
    if isinstance(router_results, list):
        result = router_results[0]
        result_text = result.raw if hasattr(result, 'raw') else str(result)
    else:
        result_text = router_results.raw if hasattr(router_results, 'raw') else str(router_results)

    task_type = result_text.strip().lower()
    return task_type


def execute_trip_task(task_type: str, trip_request: TripRequest):
    """Execute the trip planning task with the appropriate agent"""

    # Prepare the detailed request description
    request_description = f"""
    Plan a trip to {trip_request.destination} for {trip_request.duration}.
    Budget: {trip_request.budget or 'Not specified'}
    Interests: {trip_request.interests or 'General sightseeing'}
    Travel Style: {trip_request.travel_style or 'Standard'}
    """

    # Select agent and customize task based on type
    if "research" in task_type:
        agent = destination_researcher
        agent_label = "Destination Research Specialist"
        expected_output = "Comprehensive destination research including weather, attractions, culture, and practical travel information."

    elif "itinerary" in task_type:
        agent = itinerary_planner
        agent_label = "Travel Itinerary Planner"
        expected_output = "Detailed day-by-day itinerary with timings, activities, and logistics."

    elif "budget" in task_type:
        agent = budget_advisor
        agent_label = "Travel Budget Advisor"
        expected_output = "Detailed budget breakdown with cost estimates and money-saving tips."

    elif "local" in task_type:
        agent = local_expert
        agent_label = "Local Experience Curator"
        expected_output = "Authentic local experiences, restaurant recommendations, and cultural activities."

    else:
        # Default to itinerary planning
        agent = itinerary_planner
        agent_label = "Travel Itinerary Planner"
        expected_output = "Comprehensive travel plan for the requested destination."

    # Create and execute the task
    work_task = Task(
        description=request_description,
        agent=agent,
        expected_output=expected_output
    )

    worker_crew = Crew(
        agents=[agent],
        tasks=[work_task],
        process=Process.sequential,
        verbose=True
    )

    work_results = worker_crew.kickoff()

    # Extract the result text
    if isinstance(work_results, list):
        output = work_results[0].raw if hasattr(work_results[0], 'raw') else str(work_results[0])
    else:
        output = work_results.raw if hasattr(work_results, 'raw') else str(work_results)

    return output, agent_label, task_type


# Flow-based routing (Alternative approach)
class TripPlanningState(BaseModel):
    request: TripRequest
    route: str = ""
    result: str = ""


class TripPlanningFlow(Flow[TripPlanningState]):
    def __init__(self, trip_request: TripRequest):
        super().__init__(state=TripPlanningState(request=trip_request))

    @start()
    def analyze_request(self):
        print(f"🧳 Processing trip request for {self.state.request.destination}")

    @router(analyze_request)
    def decide_route(self):
        request = self.state.request
        interests = (request.interests or "").lower()

        if "budget" in interests or request.budget:
            self.state.route = "budget"
            return "budget"
        elif "food" in interests or "restaurant" in interests or "local" in interests:
            self.state.route = "local"
            return "local"
        elif "itinerary" in interests or "plan" in interests:
            self.state.route = "itinerary"
            return "itinerary"
        else:
            self.state.route = "research"
            return "research"

    @listen("research")
    def run_research(self):
        task = Task(
            description=f"Research {self.state.request.destination} for a {self.state.request.duration} trip. Include weather, attractions, culture, and practical information.",
            expected_output="Comprehensive destination research report",
            agent=destination_researcher,
        )
        crew = Crew(agents=[destination_researcher], tasks=[task], process=Process.sequential, verbose=True)
        result = crew.kickoff()
        self.state.result = result.raw if hasattr(result, 'raw') else str(result)

    @listen("itinerary")
    def run_itinerary_planning(self):
        task = Task(
            description=f"Create a detailed itinerary for {self.state.request.destination} for {self.state.request.duration}. Budget: {self.state.request.budget or 'flexible'}. Interests: {self.state.request.interests or 'general'}.",
            expected_output="Day-by-day travel itinerary with activities and timings",
            agent=itinerary_planner,
        )
        crew = Crew(agents=[itinerary_planner], tasks=[task], process=Process.sequential, verbose=True)
        result = crew.kickoff()
        self.state.result = result.raw if hasattr(result, 'raw') else str(result)

    @listen("budget")
    def run_budget_planning(self):
        task = Task(
            description=f"Create a budget breakdown for a {self.state.request.duration} trip to {self.state.request.destination}. Target budget: {self.state.request.budget or 'provide options'}.",
            expected_output="Detailed budget breakdown with cost estimates",
            agent=budget_advisor,
        )
        crew = Crew(agents=[budget_advisor], tasks=[task], process=Process.sequential, verbose=True)
        result = crew.kickoff()
        self.state.result = result.raw if hasattr(result, 'raw') else str(result)

    @listen("local")
    def run_local_planning(self):
        task = Task(
            description=f"Recommend authentic local experiences, restaurants, and cultural activities for {self.state.request.destination}. Duration: {self.state.request.duration}.",
            expected_output="Local experiences and restaurant recommendations",
            agent=local_expert,
        )
        crew = Crew(agents=[local_expert], tasks=[task], process=Process.sequential, verbose=True)
        result = crew.kickoff()
        self.state.result = result.raw if hasattr(result, 'raw') else str(result)


# API Routes
@app.get("/")
async def root():
    return {"message": "Trip Planner API is running!"}


@app.post("/plan-trip", response_model=TripResponse)
async def plan_trip(trip_request: TripRequest):
    """Plan a trip using agent routing"""
    try:
        # Create request description for routing
        request_text = f"Plan a trip to {trip_request.destination} for {trip_request.duration}"
        if trip_request.interests:
            request_text += f" with interests in {trip_request.interests}"

        # Route the request
        router_output = route_trip_request(request_text, router_agent)
        task_type = determine_task_type(router_output)

        # Execute the task
        result, agent_label, route = execute_trip_task(task_type, trip_request)

        return TripResponse(
            agent_type=agent_label,
            result=result,
            route_taken=route
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/plan-trip-flow", response_model=TripResponse)
async def plan_trip_flow(trip_request: TripRequest):
    """Plan a trip using Flow routing"""
    try:
        # Use asyncio to run the flow
        flow = TripPlanningFlow(trip_request=trip_request)
        await asyncio.get_event_loop().run_in_executor(None, flow.kickoff)

        return TripResponse(
            agent_type=f"Flow - {flow.state.route.title()} Specialist",
            result=flow.state.result,
            route_taken=flow.state.route
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "trip-planner-backend"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
