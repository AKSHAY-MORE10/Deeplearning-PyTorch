from agno.agent import Agent
from agno.models.groq import Groq
from agno.tools.websearch import WebSearchTools
from dotenv import load_dotenv

load_dotenv()

agent = Agent(
    model=Groq(id="qwen/qwen3-32b"),
    tools=[WebSearchTools(backend="brave")],
    markdown=True,
)

agent.print_response("What's happening in France?")