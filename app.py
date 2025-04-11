import streamlit as st
from agents_ia.chat import ChatAgent
from agents_ia.embedding import EmbeddingProcessor
from agents_ia.loader import CustomLoader
from agents_ia.memory import add_system_message
import os

# Define as configurações da página do Streamlit
st.set_page_config(page_title="Chat com LLaMA3", page_icon="🐉")

# ---------------------- Inicialização de estado da sessão ----------------------

# Garante que o estado da sessão está definido
if "session_id" not in st.session_state:
    st.session_state.session_id = "1"
if "chat_agent" not in st.session_state:
    st.session_state.chat_agent = {}
if "chat_histories" not in st.session_state:
    st.session_state.chat_histories = {}
if "embedded_files" not in st.session_state:
    st.session_state.embedded_files = {}
if "answer" not in st.session_state:
    st.session_state.answer = {}

# Converte session_id para string e inicializa objetos por sessão, se necessário
st.session_state.session_id = str(st.session_state.session_id)
session_id = st.session_state.session_id
if session_id not in st.session_state.chat_agent:
    st.session_state.chat_agent[session_id] = ChatAgent()
if session_id not in st.session_state.embedded_files:
    st.session_state.embedded_files[session_id] = set()
if session_id not in st.session_state.chat_histories:
    st.session_state.chat_histories[session_id] = []

# ---------------------- Barra lateral para múltiplas sessões ----------------------

st.sidebar.title("Sessões de Chat")
st.sidebar.markdown("---")

# Campo para criar nova sessão
new_session_id = st.sidebar.text_input("Nova sessão ID", value="")
if st.sidebar.button("Criar nova sessão"):
    new_session_id = str(new_session_id).strip()
    if new_session_id and new_session_id not in st.session_state.chat_histories:
        st.session_state.chat_histories[new_session_id] = []
        st.session_state.session_id = new_session_id


# Função para trocar de sessão
def trocar_sessao(sid):
    st.session_state.session_id = sid


# Lista as sessões ativas como botões na sidebar
st.sidebar.markdown("# Sessões Ativas:")
for sid in st.session_state.chat_histories.keys():
    sid_str = str(sid)
    st.sidebar.button(
        f"Sessão {sid_str}",
        key=f"botao_sessao_{sid_str}",
        on_click=trocar_sessao,
        args=(sid_str,),
    )

# ---------------------- Interface principal ----------------------

# Tabs para alternar entre chat e debug
tabs = st.tabs(["  Assistente Virtual  ", " Debug/Info "])

# Histórico de chat e respostas por sessão
current_session_id = st.session_state.session_id
chat_history = st.session_state.chat_histories.get(current_session_id, [])
if current_session_id not in st.session_state.answer:
    st.session_state.answer[current_session_id] = []
if current_session_id not in st.session_state.embedded_files:
    st.session_state.embedded_files[current_session_id] = set()

# Campo de entrada com suporte a upload de múltiplos arquivos
prompt = st.chat_input(
    "Digite sua mensagem ou insira um arquivo PDF ou CSV",
    accept_file="multiple",
    file_type=["pdf", "csv"],
)

# ---------------------- Aba: Assistente Virtual ----------------------
with tabs[0]:
    # Mostra o histórico de mensagens
    for msg in chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    # Processamento de arquivos PDF ou CSV enviados pelo usuário
    if prompt and prompt["files"]:
        for uploaded_file in prompt["files"]:
            filename = uploaded_file.name
            ext = filename.split(".")[-1].lower()

            # Ignora arquivos já embeddados
            if filename in st.session_state.embedded_files[session_id]:
                continue

            with st.spinner("Vetorizando o arquivo..."):
                loader = CustomLoader(file=uploaded_file, filename=filename)
                docs = loader._load()
                uploaded_file.close()

                # Caso seja PDF, cria o retriever com embeddings e atualiza o agente
                if ext == "pdf":
                    processor = EmbeddingProcessor(data=docs, session_id=session_id)
                    retriever = processor.create_retriever()
                    st.session_state.chat_agent[session_id].trocar_para_rag(retriever)
                    st.session_state.embedded_files[session_id].add(filename)

                    add_system_message(
                        session_id,
                        f"Voce agora possui conhecimento sobre o arquivo PDF (que podem ser referidos como textos, artigos, etc) de nome: '{filename}'! \nResponda perguntas sobre o conteúdo desse arquivo. \nSe não souber a resposta, seja honesto e proponha caminhos para buscar a informação. \nSe tiver mais de um arquivo, use o nome do arquivo para referenciar o conteúdo que deseja consultar.",
                    )

                    st.session_state.chat_histories[session_id].append(
                        {
                            "role": "system",
                            "content": f"Arquivo '{filename}' embeddado com sucesso!",
                        }
                    )
                    st.success(f"PDF '{filename}' embeddado com sucesso!")

                # Caso seja CSV, processa com ferramenta de análise de dados
                elif ext == "csv":
                    st.session_state.chat_agent[session_id].load_dataframe_tools(
                        df=docs
                    )
                    add_system_message(
                        session_id,
                        f"Voce agora possui conhecimento sobre o arquivo CSV de nome: '{filename}'! \nResponda perguntas sobre o conteúdo desse arquivo utilizando apenas python e pandas. \nSe não souber a resposta, seja honesto e proponha caminhos para buscar a informação. \nSe tiver mais de um arquivo, use o nome do arquivo para referenciar o conteúdo que deseja consultar.",
                    )

                    st.session_state.chat_histories[session_id].append(
                        {
                            "role": "system",
                            "content": f"Arquivo '{filename}' carregado com sucesso!",
                        }
                    )
                    st.success(f"CSV '{filename}' processado com sucesso!")
            st.session_state.carregando_arquivo = False
    # Processa a mensagem textual enviada no prompt
    if prompt and prompt.text:
        st.chat_message("user").markdown(prompt.text)

        with st.chat_message("assistant"):
            with st.spinner("Pensando..."):
                answer = st.session_state.chat_agent[current_session_id].responder(
                    prompt.text, session_id=current_session_id
                )
                st.session_state.answer[current_session_id] = answer

                # Extrai o texto da resposta (pode ser string ou objeto Message)
                answer_text = (
                    answer.get("answer") if isinstance(answer, dict) else answer.content
                )
                st.markdown(answer_text)

        # Atualiza histórico da sessão
        st.session_state.chat_histories[current_session_id].append(
            {"role": "user", "content": prompt.text}
        )
        st.session_state.chat_histories[current_session_id].append(
            {"role": "assistant", "content": answer_text}
        )

# ---------------------- Aba: Debug/Info ----------------------
with tabs[1]:
    st.subheader("Sessão atual:")
    st.write("session_id:", current_session_id)
    st.write(
        "Arquivos embeddados:", st.session_state.embedded_files[current_session_id]
    )
    st.write("Conteudo Resposta LLM:", st.session_state.answer[current_session_id])
