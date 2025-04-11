from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain.chat_models.base import BaseChatModel
import os

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")


class LocalLLM:
    # This class is used to load a local LLM model using the chat_models library (Ollama).
    def __init__(self, model_name="llama3.2", temperature=0.75):
        self.model_name = model_name
        self.temperature = temperature
        self.llm = self.get_model()

    def get_model(self) -> BaseChatModel:
        try:
            return ChatOllama(
                model=self.model_name,
                temperature=self.temperature,
                stream=True,
                base_url=OLLAMA_URL,
            )
        except Exception as e:
            print(
                f"Error loading model {self.model_name}, make sure you have installed the model and Ollama is running. \nError: {e}"
            )
            raise e


class EmbeddingLLM:
    # This class is used to load a local LLM embedding model using the langchain library (Ollama).
    def __init__(self, model_name="mxbai-embed-large"):
        self.model_name = model_name
        self.embedding_llm = self.get_model()

    def get_model(self):
        try:
            return OllamaEmbeddings(
                model=self.model_name,
                base_url=OLLAMA_URL,
            )
        except Exception as e:
            print(
                f"Error loading model {self.model_name}, make sure you have installed the model and Ollama is running. \nError: {e}"
            )
            raise e
