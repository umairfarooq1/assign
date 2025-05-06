import os

import _asyncio
from agents import Agent, AsyncOpenAI, OpenAIChatCompletionsModel, Runner
from agents.run import RunConfig
from dotenv import load_dotenv
from rich import print

load_dotenv()

api_key = os.getenv("gemini_api_key")


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

    result = Runner.run_sync(agent, "What is capital of Pakistan?", run_config=config)

    print("\n")
    print(result.final_output)