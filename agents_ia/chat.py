from langchain_core.runnables import RunnableLambda, Runnable
from typing import List, Iterator
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from LLM.local_llm import LocalLLM
from agents_ia.embedding import EmbeddingProcessor
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

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


# Função para mostrar o histórico de mensagens (debug)
def mostrar_historico(session_id: str):
    historico = get_session_history(session_id)
    if len(historico.messages) > 0:
        print(f"Histórico de Mensagens da Sessão {session_id}:")
        for i, mensagem in enumerate(historico.messages, start=1):
            print(f"Mensagem {i} - {mensagem.type}:")
            print(mensagem.content)
            print("-" * 30)
    else:
        print(f"Sessão {session_id} não tem mensagens")


class ChatAgent:
    def __init__(self, retriever=None):
        """
        Se retriever for passado, usa RAG.
        Caso contrário, usa apenas o LLM com histórico simples.
        """
        self.local_llm = LocalLLM().llm

        if retriever:
            self.rag_chain = self.build_rag_chain(retriever)
            self.chat_with_history = RunnableWithMessageHistory(
                self.rag_chain,
                get_session_history,
                input_messages_key="input",
                history_messages_key="chat_history",
                output_messages_key="answer",
            )
        else:
            # fallback: só LLM com histórico
            self.chat_with_history = RunnableWithMessageHistory(
                runnable=RunnableLambda(lambda x: self.local_llm.invoke(x["messages"])),
                get_session_history=get_session_history,
                input_messages_key="messages",
            )

    def build_rag_chain(self, retriever):
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "Dado um histórico de conversa e a pergunta mais recente do usuário,"
                        "que pode fazer referência ao contexto do histórico,"
                        "formule uma pergunta independente que possa ser entendida"
                        "sem depender do histórico da conversa. NÃO responda à pergunta,"
                        "apenas reformule-a se necessário; caso contrário, retorne como está."
                    ),
                ),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        history_aware_retriever = create_history_aware_retriever(
            self.local_llm, retriever, contextualize_q_prompt
        )

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "Você é um assistente para tarefas de perguntas e respostas. "
                        "Use os seguintes trechos de contexto recuperado para responder "
                        "a pergunta. Se você não souber a resposta, diga que não sabe. "
                        "Use no máximo três frases e mantenha a resposta concisa."
                        "\n\n"
                        "{context}"
                    ),
                ),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        question_answer_chain = create_stuff_documents_chain(self.local_llm, qa_prompt)

        return create_retrieval_chain(history_aware_retriever, question_answer_chain)

    def responder(self, pergunta: str, session_id: int = 1):
        """
        Se usando RAG, input é 'input'
        Caso contrário, input é 'messages'
        """
        if hasattr(self, "rag_chain"):
            entrada = {"input": pergunta}
        else:
            entrada = {"messages": [HumanMessage(content=pergunta)]}

        resposta = self.chat_with_history.invoke(
            entrada,
            config={"configurable": {"session_id": session_id}},
        )
        mostrar_historico(session_id)
        return resposta
