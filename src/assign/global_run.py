import os

from agents import Agent, AsyncOpenAI, OpenAIChatCompletionsModel, Runner
from agents.run import RunConfig
from dotenv import load_dotenv
from rich import print

load_dotenv()

api_key = os.getenv("gemini_api_key")
name = os.getenv("name")


external_client = AsyncOpenAI(
    api_key=api_key, base_url="https://generativelanguage.googleapis.com/v1beta/"
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client,
)


def run_global():
    agent = Agent(
        name="Assistant",
        instructions="A helpful assistant that can answer questions and provide information.",
    )

    config = RunConfig(
        model=model,
        tracing_disabled=True,
    )

    result = Runner.run_sync(agent, "What are the latest tecniques to investigation of police cases?", run_config=config)

    print("\n")
    print(result.final_output)
    
    with open("README.md", "a", encoding="utf-8") as readme_file:
        readme_file.write("\n## Agent Response:\n")
        readme_file.write(result.final_output + "\n")
