from langchain_ollama import ChatOllama
from langchain.chat_models.base import BaseChatModel


class LocalLLM:
    # This class is used to load a local LLM model using the chat_models library (Ollama).
    def __init__(self, model_name="llama3.2", temperature=0.75):
        self.model_name = model_name
        self.temperature = temperature
        self.llm = self.get_model()

    def get_model(self) -> BaseChatModel:
        return ChatOllama(
            model=self.model_name, temperature=self.temperature, stream=True
        )
