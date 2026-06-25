"""Central LLM factory — single construction site for the Anthropic client."""
import os
from langchain_anthropic import ChatAnthropic


def get_llm(temperature: float = 0.0, max_tokens: int = 1000) -> ChatAnthropic:
    return ChatAnthropic(
        model="claude-sonnet-4-6",
        max_tokens=max_tokens,
        temperature=temperature,
        api_key=os.environ["ANTHROPIC_API_KEY"],
    )
