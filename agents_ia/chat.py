from langchain_core.runnables import RunnableLambda, Runnable
from typing import List, Iterator
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
    AIMessage,
    ToolMessage,
)
from LLM.local_llm import LocalLLM
from agents_ia.embedding import EmbeddingProcessor
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import pandas as pd
from langchain_experimental.tools import PythonAstREPLTool
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_core.output_parsers.openai_tools import JsonOutputKeyToolsParser

# Armazena o histórico por sessão
store = {}


# Função para buscar (ou criar) o histórico da sessão
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
        store[session_id].add_message(
            SystemMessage(
                content=(
                    """
                    Você é um assistente de IA útil, educado e claro. 
                    Responda sempre em português, a menos que o usuário solicite outro idioma. 
                    Seja conciso e objetivo, com foco em ajudar com dúvidas técnicas.
                    Seu papel é auxiliar o usuário de forma contextual, com base em documentos fornecidos e ferramentas disponíveis.  
                    Você deve:
                        1. Analisar os documentos carregados pelo usuário e extrair informações relevantes.
                        2. Gerar planos de ação para resolver problemas apresentados pelo usuário, com clareza e lógica.
                        3. Responder perguntas com base no conhecimento extraído dos documentos enviados.

                    Se uma ferramenta for usada, utilize o resultado dela para compor sua resposta final, explicando de forma compreensível.  
                    Não repita ferramentas ou chame novas ferramentas se já houver um resultado disponível.

                    Se não souber a resposta, seja honesto e proponha caminhos para buscar a informação.
                    """
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
    def __init__(self, retriever=None, df=None):
        """
        Se retriever for passado, usa RAG.
        Caso contrário, usa apenas o LLM com histórico simples.
        """
        self.local_llm = LocalLLM().llm
        self._tipo_runnable = "llm"

        if retriever:
            self._tipo_runnable = "rag"
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
            self._tipo_runnable = "llm"
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

        if self._tipo_runnable == "csv":
            # Etapa 1 - Envia a pergunta para o modelo
            print(f"CSV")
            entrada = {"input": pergunta}
            print(f"Entrada: {entrada}")
            resposta_inicial = self.chat_with_history.invoke(
                entrada,
                config={"configurable": {"session_id": session_id}},
            )
            print(f"Resposta inicial: {resposta_inicial}")
            # Etapa 2 - Verifica se o modelo chamou uma ferramenta
            tool_calls = (
                resposta_inicial.tool_calls
                if hasattr(resposta_inicial, "tool_calls")
                else []
            )
            print(f"Tool calls: {tool_calls}")
            if tool_calls:
                tool_call = tool_calls[0]
                tool_name = tool_call["name"]
                tool_args = tool_call.get("args", {})
                tool_id = tool_call["id"]

                # Etapa 3 - Executa a ferramenta correspondente
                tool_result = None
                if tool_name == "python_repl_ast":
                    query = tool_args.get("query", "")
                    tool_result = self.tool_csv.run(query)
                elif tool_name.startswith("df"):
                    query = tool_name
                    tool_result = self.tool_csv.run(query)
                print(f"Resultado da ferramenta: {tool_result}")

                # Etapa 4 - Monta o histórico manual com ToolMessage
                historico = get_session_history(session_id)
                historico.add_message(HumanMessage(content=pergunta))
                historico.add_message(
                    AIMessage(
                        content="",
                        additional_kwargs={"tool_calls": [tool_call]},
                    )
                )
                historico.add_message(
                    ToolMessage(tool_call_id=tool_id, content=str(tool_result))
                )

                # Etapa 5 - Envia de volta ao modelo para gerar resposta final
                resposta_final = self.local_llm.invoke(
                    historico.messages,
                    config={"configurable": {"session_id": session_id}},
                )

                # mostrar_historico(session_id)
                print(f"\nResposta final: {resposta_final}")
                return resposta_final

            # Se não houve tool_call, retorno direto
            # mostrar_historico(session_id)
            return resposta_inicial

        if self._tipo_runnable == "llm":
            entrada = {"messages": [HumanMessage(content=pergunta)]}
        else:
            entrada = {"input": pergunta}

        resposta = self.chat_with_history.invoke(
            entrada,
            config={"configurable": {"session_id": session_id}},
        )
        # mostrar_historico(session_id)
        return resposta

    def load_dataframe_tools(self, df):
        self._tipo_runnable = "csv"
        self.tool_csv = PythonAstREPLTool(locals={"df": df})
        llm_tool = self.local_llm.bind_tools(
            [self.tool_csv], tool_choice="python_repl_ast"
        )
        system = f"""
            Você tem acesso ao DataFrame `df` para responder perguntas sobre os dados.
            - Utilize apenas o DataFrame `df` para responder às perguntas.
            - Use SOMENTE Python com pandas.
            - Sempre chame a ferramenta apenas com o código necessário.
            - Depois de obter a resposta da ferramenta, retorne-a ao usuário."
            """
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    system,
                ),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        chain = prompt | llm_tool
        self.chat_with_history = RunnableWithMessageHistory(
            runnable=chain,
            get_session_history=get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )
