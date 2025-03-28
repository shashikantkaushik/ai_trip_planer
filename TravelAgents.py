from crewai import Agent, Crew, Task, Process
from TravelTools import search_web_tool
from crewai import LLM
import time

# Initialize Groq LLM
llm = LLM(
    model="groq/llama3-70b-8192",
    api_key="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",
    temperature=0.3,
    max_tokens=1000
)


# Define Agents
def create_agent(role, goal, backstory):
    return Agent(
        role=role,
        goal=goal,
        backstory=backstory,
        tools=[search_web_tool],
        llm=llm,
        verbose=True,  # Boolean value
        max_iter=3,
        allow_delegation=False
    )


guide_expert = create_agent(
    role="Rome Local Guide",
    goal="Find best sightseeing spots and authentic food",
    backstory="Roman native with 10+ years guiding tourists"
)

logistics_expert = create_agent(
    role="Travel Logistics Expert",
    goal="Provide visa, transport and budget info",
    backstory="Former travel agent specializing in Europe"
)

planner_expert = create_agent(
    role="Trip Planner",
    goal="Create perfect 1-day itineraries",
    backstory="Professional travel itinerary designer"
)


# Define Tasks
def create_task(description, agent, expected_output, context=[]):
    return Task(
        description=description,
        agent=agent,
        expected_output=expected_output,
        context=context
    )


research_task = create_task(
    description="Research top attractions and restaurants in Rome for an Indian traveler on March 29, 2025",
    agent=guide_expert,
    expected_output="List of 5 attractions and 3 restaurants with descriptions"
)

logistics_task = create_task(
    description="Provide visa requirements, transport options and budget estimates",
    agent=logistics_expert,
    expected_output="Clear visa info, transport recommendations, and cost breakdown"
)

planning_task = create_task(
    description="Create detailed 1-day itinerary with time slots and meal suggestions",
    agent=planner_expert,
    expected_output="Hour-by-hour schedule with travel times",
    context=[research_task, logistics_task]
)


# Create and Run Crew with Error Handling
def run_crew():
    crew = Crew(
        agents=[guide_expert, logistics_expert, planner_expert],
        tasks=[research_task, logistics_task, planning_task],
        process=Process.sequential,
        verbose=True  # Changed from 2 to True
    )

    try:
        result = crew.kickoff()
        return result
    except Exception as e:
        print(f"Error: {e}")
        return """
        ⚠️ API Limit Reached - Sample Itinerary:

        09:00 🏛️ Colosseum Tour
        11:00 🏟️ Roman Forum
        13:00 🍝 Lunch at Roscioli (Carbonara)
        15:00 ⛲ Trevi Fountain & Pantheon
        18:00 🏛️ Vatican City
        20:00 🍕 Dinner at Da Baffetto

        💡 Try again in 20 seconds for full details
        """


print("⏳ Planning your trip...")
time.sleep(2)
itinerary = run_crew()
print("\n🌟 Your Personalized Itinerary:\n", itinerary)