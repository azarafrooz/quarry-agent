from transformers import pipeline
import torch
from ..agent import AbstractPythonCodeAgent
from typing import Any, List
from ..utils import CountCalls


class HfAgent(AbstractPythonCodeAgent):
    counter = CountCalls()

    def __init__(self, config=None):
        model_name_or_path = config.get("model_name_or_path", None)
        device = config.get("device", None)
        self.pipe = pipeline(
           "text-generation",
           model=model_name_or_path,
           device=device,  # replace with "mps" to run on a Mac device
           model_kwargs={"torch_dtype": torch.bfloat16},
          trust_remote_code=True,
        )

    def system_message(self, message: str) -> Any:
        return {"role": "system", "content": message}

    def user_message(self, message: str) -> Any:
        return {"role": "user", "content": message}

    def assistant_message(self, message: str) -> Any:
        return {"role": "assistant", "content": message}

    @counter
    def submit_prompt(self, prompt, **kwargs) -> str:
        outputs = self.pipe(prompt, **kwargs)
        response = outputs[0]["generated_text"][-1]["content"]

        return response
