from langchain_community.llms import CTransformers
import os


class LocalLLM:
    # This class is used to load a local LLM model using the CTransformers library.
    def __init__(
        self,
        model_path: str = "llm/models",
        model_file: str = "llama-2-7b-chat.Q4_K_M.gguf",
    ):
        full_path = os.path.join(model_path, model_file)

        self.llm = CTransformers(
            model=full_path,
            model_type="llama",
            config={
                "temperature": 0.7,
                "repetition_penalty": 1.1,
                "context_length": 512,
                "stream": True,
                "stop": ["Usuário:", "User:", "\n\n"],
            },
        )

    def get_model(self):
        return self.llm
