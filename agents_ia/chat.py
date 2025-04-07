from langchain_core.runnables import RunnableLambda, Runnable
from typing import List, Iterator
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from LLM.local_llm import LocalLLM

# Armazena o histórico por sessão
store = {}


# Função para buscar (ou criar) o histórico da sessão
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
        store[session_id].add_message(
            SystemMessage(
                content=(
                    "Você é um assistente de IA útil, educado e claro. "
                    "Responda sempre em português, a menos que o usuário peça outro idioma. "
                    "Seja conciso e objetivo. Ajude com dúvidas técnicas, principalmente sobre Python."
                )
            )
        )
    return store[session_id]


"""
class StreamRunnable(Runnable[List[BaseMessage], Iterator[BaseMessage]]):
    def __init__(self, llm):
        self.llm = llm

    def invoke(
        self, messages: List[BaseMessage], config=None, **kwargs
    ) -> Iterator[BaseMessage]:
        # Chama o LLM local com as mensagens formatadas
        return self.llm.invoke(messages_to_prompt(messages))

    def stream(
        self, messages: List[BaseMessage], config=None, **kwargs
    ) -> Iterator[BaseMessage]:
        return self.llm.stream(messages)
"""


class ChatAgent:
    # Classe para o agente de chat que utiliza um LLM local
    # O agente é responsável por gerenciar o histórico de mensagens e formatar as mensagens para o LLM Local
    def __init__(self):
        self.local_llm = LocalLLM().llm

        # Adiciona suporte a histórico
        self.chat_with_history = RunnableWithMessageHistory(
            runnable=RunnableLambda(lambda x: self.local_llm.invoke(x["messages"])),
            get_session_history=get_session_history,
            input_messages_key="messages",
        )

    def responder(self, pergunta: str, session_id: int = 1):
        mensagens: list[BaseMessage] = [HumanMessage(content=pergunta)]
        resposta = self.chat_with_history.invoke(
            {"messages": mensagens},
            config={"configurable": {"session_id": session_id}},
        )
        return resposta
