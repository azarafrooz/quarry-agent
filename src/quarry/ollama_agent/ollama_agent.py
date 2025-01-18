from httpx import Timeout
from ..agent import AbstractPythonCodeAgent
from ..exceptions import DependencyError
from ..utils import CountCalls


class OllamaAgent(AbstractPythonCodeAgent):
    counter = CountCalls()

    def __init__(self, config=None):
        try:
            ollama = __import__("ollama")
        except ImportError:
            raise DependencyError(
                "You need to install required dependencies to execute this method, run command:"
                " \npip install ollama"
            )

        if not config:
            raise ValueError("config must contain at least Ollama model")
        if "model" not in config.keys():
            raise ValueError("config must contain at least Ollama model")
        self.host = config.get("ollama_host", "http://localhost:11434")
        self.model = config["model"]
        if ":" not in self.model:
            self.model += ":latest"

        self.ollama_client = ollama.Client(self.host, timeout=Timeout(240.0))
        self.keep_alive = config.get("keep_alive", None)
        self.ollama_options = config.get("options", {})
        self.num_ctx = self.ollama_options.get("num_ctx", 8192)
        self.__pull_model_if_ne(self.ollama_client, self.model)

    @staticmethod
    def __pull_model_if_ne(ollama_client, model):
        model_response = ollama_client.list()
        model_lists = [
            model_element["model"] for model_element in model_response.get("models", [])
        ]
        if model not in model_lists:
            ollama_client.pull(model)

    def system_message(self, message: str) -> any:
        return {"role": "system", "content": message}

    def user_message(self, message: str) -> any:
        return {"role": "user", "content": message}

    def assistant_message(self, message: str) -> any:
        return {"role": "assistant", "content": message}

    @counter
    def submit_prompt(self, prompt, **kwargs) -> str:
        response_dict = self.ollama_client.chat(
            model=self.model,
            messages=prompt,
            stream=False,
            options=self.ollama_options,
            keep_alive=self.keep_alive,
        )

        return response_dict["message"]["content"]
