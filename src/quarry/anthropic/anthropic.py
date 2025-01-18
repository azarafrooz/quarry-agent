from ..agent import AbstractPythonCodeAgent
from typing import Any
from anthropic import Anthropic

from ..utils import CountCalls


class AnthropicAgent(AbstractPythonCodeAgent):
    counter = CountCalls()

    def __init__(self, config=None):
        self.model_name_or_path = config.get("model_name_or_path", "claude-3-5-haiku")
        self.client = Anthropic(api_key=config.get("anthropic_api_key", None))

    def system_message(self, message: str) -> Any:
        return {"role": "system", "content": message}

    def user_message(self, message: str) -> Any:
        content = [
            {"type": "text", "text": message, "cache_control": {"type": "ephemeral"}}
        ]
        return {"role": "user", "content": content}

    def assistant_message(self, message: str) -> Any:
        return {"role": "assistant", "content": message.strip()}

    @counter
    def submit_prompt(self, prompt, **kwargs) -> str:
        response = self.client.beta.messages.create(
            model=self.model_name_or_path, messages=prompt, **kwargs
        )
        # # Print cache performance metrics
        # print(f"Input tokens: {response.usage.input_tokens}")
        # print(f"Output tokens: {response.usage.output_tokens}")
        # print(f"Cache creation input tokens: {response.usage.cache_creation_input_tokens}")
        # print(f"Cache read input tokens: {response.usage.cache_read_input_tokens}")
        return response.content[0].text
