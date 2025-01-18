from ..agent import AbstractPythonCodeAgent
from typing import Any
from openai import OpenAI
from ..utils import CountCalls


class OpenAIAgent(AbstractPythonCodeAgent):
    counter = CountCalls()

    def __init__(self, config=None):
        self.model_name_or_path = config.get("model_name_or_path", "gpt-4o-mini")
        self.client = OpenAI(api_key=config.get("openai_api_key", None))

    def system_message(self, message: str) -> Any:
        return {"role": "system", "content": message}

    def user_message(self, message: str) -> Any:
        return {"role": "user", "content": message}

    def assistant_message(self, message: str) -> Any:
        return {"role": "assistant", "content": message}

    @counter
    def submit_prompt(self, prompt, **kwargs) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name_or_path, messages=prompt, stop=None, **kwargs
        )

        return response.choices[0].message.content
