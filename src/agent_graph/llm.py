"""Central LLM factory — single construction site for the Anthropic client."""
import os
from langchain_anthropic import ChatAnthropic


def get_llm(temperature: float = 0.0) -> ChatAnthropic:
    return ChatAnthropic(
        model="claude-sonnet-4-6",
        max_tokens=1000,
        temperature=temperature,
        api_key=os.environ["ANTHROPIC_API_KEY"],
    )
