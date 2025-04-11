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
from agents_ia.memory import get_session_history
from langchain_core.output_parsers import StrOutputParser


def mostrar_historico(session_id: str):
    """
    Exibe no console o histórico de mensagens de uma sessão específica.

    session_id: string que identifica unicamente uma sessão.

    return: None
    """
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
        Inicializa o agente de chat, com ou sem RAG.

        retriever: objeto opcional para recuperação de contexto com embeddings (RAG).
        df: dataframe opcional (não utilizado diretamente na inicialização).

        return: None
        """
        self.local_llm = LocalLLM().llm
        self._tipo_runnable = "llm"

        default_prompt = ChatPromptTemplate.from_messages(
            [
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        chain = default_prompt | self.local_llm
        self.chat_with_history = RunnableWithMessageHistory(
            runnable=chain,
            get_session_history=get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
        )

    def trocar_para_rag(self, retriever):
        """
        Alterna o agente para o modo RAG (retrieval augmented generation),
        com encadeamento de reformulação de perguntas e busca de contexto.

        retriever: objeto de busca de documentos por similaridade vetorial.

        return: None
        """
        self._tipo_runnable = "rag"
        self.rag_chain = self.build_rag_chain(retriever)
        self.chat_with_history = RunnableWithMessageHistory(
            self.rag_chain,
            get_session_history=get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

    def build_rag_chain(self, retriever):
        """
        Cria um encadeamento de RAG que contextualiza perguntas com base no histórico e documentos relevantes.

        retriever: objeto para recuperação de documentos via embeddings.

        return: Runnable: cadeia de execução que realiza RAG.
        """
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "Você é um modelo que reformula perguntas de forma independente com base no histórico de chat. "
                        "Receba a conversa até agora e uma nova pergunta, e devolva uma versão clara e autossuficiente da pergunta. "
                        "Considere também o histórico anterior para manter coerência."
                        "NÃO responda. Reformule apenas."
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
                        "Se a pergunta não estiver clara, faça perguntas adicionais para obter mais detalhes. "
                        "Use o contexto abaixo para responder perguntas:\n\n{context}"
                    ),
                ),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        question_answer_chain = create_stuff_documents_chain(self.local_llm, qa_prompt)

        return create_retrieval_chain(history_aware_retriever, question_answer_chain)

    def responder(self, pergunta: str, session_id: str):
        """
        Envia uma pergunta ao modelo, considerando o tipo atual de execução (simples, RAG ou CSV).

        pergunta: string com a pergunta do usuário.
        session_id: identificador da sessão de chat.

        return:
            - resposta (string ou objeto AIMessage) gerada pelo modelo.
        """
        if self._tipo_runnable == "csv":
            entrada = {"input": pergunta}
            resposta_inicial = self.chat_with_history.invoke(
                entrada,
                config={"configurable": {"session_id": session_id}},
            )

            tool_calls = (
                resposta_inicial.tool_calls
                if hasattr(resposta_inicial, "tool_calls")
                else []
            )
            if tool_calls:
                tool_call = tool_calls[0]
                tool_name = tool_call["name"]
                tool_args = tool_call.get("args", {})
                tool_id = tool_call["id"]

                tool_result = None
                if tool_name == "python_repl_ast":
                    query = tool_args.get("query", "")
                    tool_result = self.tool_csv.run(query)
                elif tool_name.startswith("df"):
                    query = tool_name
                    tool_result = self.tool_csv.run(query)

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

                local_llm_csv = LocalLLM(temperature=0.3).llm
                resposta_final = local_llm_csv.invoke(
                    historico.messages,
                    config={"configurable": {"session_id": session_id}},
                )

                return resposta_final

            return resposta_inicial

        entrada = {"input": pergunta}

        resposta = self.chat_with_history.invoke(
            entrada,
            config={"configurable": {"session_id": session_id}},
        )
        return resposta

    def load_dataframe_tools(self, df):
        """
        Configura ferramentas de análise de dados para interação com um DataFrame.
        Ativa o modo CSV com suporte a execução de código Python para análise de dados.

        df: pandas.DataFrame contendo os dados a serem analisados pelo agente.

        return: None
        """
        self._tipo_runnable = "csv"
        self.tool_csv = PythonAstREPLTool(locals={"df": df})
        llm_tool = self.local_llm.bind_tools(
            [self.tool_csv], tool_choice="python_repl_ast"
        )
        system = """
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
