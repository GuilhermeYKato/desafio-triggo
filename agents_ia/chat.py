from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)

from langchain_core.messages import BaseMessage, HumanMessage
from LLM.local_llm import LocalLLM

# Armazena o histórico por sessão
store = {}


# Função para buscar (ou criar) o histórico da sessão
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


# Função que converte o histórico de mensagens para um único prompt string
def messages_to_prompt(messages: list[BaseMessage]) -> str:
    prompt = "Você é um assistente de IA. Responda às perguntas do usuário com clareza e precisão em português se não for especificado pelo Usuário qual linguagem ele prefere.\n"
    for msg in messages:
        # print("msg:", msg.type, msg.content)
        role = "Usuário" if msg.type == "human" else "Assistente"
        prompt += f"{role}: {msg.content}"
    prompt += "Assistente:"
    # print("prompt: ", prompt)
    return prompt


class ChatAgent:
    # Classe para o agente de chat que utiliza um LLM local
    # O agente é responsável por gerenciar o histórico de mensagens e formatar as mensagens para o LLM Local
    def __init__(self):
        local_llm = LocalLLM().get_model()

        # Cria um RunnableLambda para o LLM local
        # O RunnableLambda é um wrapper que permite usar funções como se fossem objetos Runnable
        self.llm_with_prompt = RunnableLambda(
            lambda x: local_llm.invoke(messages_to_prompt(x["messages"]), stream=True)
        )

        # Adiciona suporte a histórico
        self.chat_with_history = RunnableWithMessageHistory(
            runnable=self.llm_with_prompt,
            get_session_history=get_session_history,
            input_messages_key="messages",
        )

    def responder(self, pergunta: str, session_id: str = "foo"):

        messages = [HumanMessage(content=pergunta)]
        return self.chat_with_history.stream(
            {"messages": messages},
            config={"configurable": {"session_id": session_id}},
        )
