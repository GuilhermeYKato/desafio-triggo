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
                    Se o usuario tiver um nome, use-o para se referir a ele.
                    Seu papel é auxiliar o usuário de forma contextual, com base em documentos fornecidos e ferramentas disponíveis.  
                    Você deve:
                        1. Analisar os documentos carregados pelo usuário e extrair informações relevantes.
                        2. Gerar planos de ação para resolver problemas apresentados pelo usuário, com clareza e lógica.
                        3. Responder perguntas com base no conhecimento extraído dos documentos enviados.

                    Se uma ferramenta for usada, utilize o resultado dela para compor sua resposta final, explicando de forma compreensível.  
                    Se não souber a resposta, seja honesto e proponha caminhos para buscar a informação.
                    """
                )
            )
        )
    return store[session_id]


def add_system_message(session_id: str, texto: str):
    """
    Adiciona uma mensagem do sistema ao histórico da sessão.
    """
    if session_id not in store:
        get_session_history(session_id)
    store[session_id].add_message(SystemMessage(content=texto))
    print(f"Mensagem do sistema adicionada na sessão {session_id}: {texto}")
