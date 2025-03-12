import os
from dotenv import load_dotenv
from agents import set_default_openai_key    

load_dotenv()
set_default_openai_key(os.getenv("OPENAI_API_KEY"))

from agents import Agent, Runner, GuardrailFunctionOutput, WebSearchTool
from pydantic import BaseModel

class HomeworkOutput(BaseModel):
    is_homework: bool
    reasoning: str

guardrail_agent = Agent(
    name="Guardrail check",
    instructions="Check if the user is asking about homework.",
    output_type=HomeworkOutput,
)

math_tutor_agent = Agent(
    name="Math Tutor",
    handoff_description="Specialist agent for math questions",
    instructions="You provide help with math problems. Explain your reasoning at each step and include examples",
)

history_tutor_agent = Agent(
    name="History Tutor",
    handoff_description="Specialist agent for historical questions",
    instructions="You provide assistance with historical queries. Explain important events and context clearly.",
)

triage_agent = Agent(
    name="Triage Agent",
    instructions="You determine which agent to use based on the user's homework question",
    handoffs=[history_tutor_agent, math_tutor_agent]
)

WeatherAgent = Agent(
    name="Weather Agent",
    instructions="You provide the weather for a given city",
    model="gpt-4o-mini",
    tools=[WebSearchTool()],
)


async def main():
    result = await Runner.run(WeatherAgent, "What is the weather in Paris?")
    print(result.final_output)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())