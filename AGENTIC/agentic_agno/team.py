from agno.agent import Agent
from agno.models.groq import Groq
from agno.team import Team
from dotenv import load_dotenv

load_dotenv()

eng_agent = Agent(
    name="English Agent",
    instructions="Always answer ONLY in English."
)

chi_agent = Agent(
    name="Chinese Agent",
    instructions="Always answer ONLY in Chinese."
)

hindi_agent = Agent(
    name="Hindi Agent",
    instructions="Always answer ONLY in Hindi."
)

team_leader = Team(
    name="Answer & Translation Team",
    members=[eng_agent, chi_agent, hindi_agent],
    model=Groq(id="qwen/qwen3-32b"),
    markdown=True,
    show_members_responses=True,
    instructions="""
    Every member agent MUST answer the user query.
    
    - English Agent answers in English
    - Chinese Agent answers in Chinese
    - Hindi Agent answers in Hindi
    
    Do NOT skip any agent.
    Combine and show all responses.
    """
)

team_leader.print_response("What is the capital of India?")