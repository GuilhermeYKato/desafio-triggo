import streamlit as st
from agents_ia.chat import ChatAgent
from agents_ia.embedding import EmbeddingProcessor
from agents_ia.loader import CustomLoader
from agents_ia.memory import add_system_message

st.set_page_config(page_title="Chat com LLaMA3", page_icon="🐉")
st.title("Chatbot com LLaMA3")

# Estado inicial para a sessão do Streamlit
# Inicializa o estado da sessão se não existir
if "session_id" not in st.session_state:
    st.session_state.session_id = "1"

if "chat_agent" not in st.session_state:
    st.session_state.chat_agent = {}

if "chat_histories" not in st.session_state:
    st.session_state.chat_histories = {}

if "embedded_files" not in st.session_state:
    st.session_state.embedded_files = {}


# Garantir que sessão atual tenha estado inicializado
st.session_state.session_id = str(st.session_state.session_id)
session_id = st.session_state.session_id
if session_id not in st.session_state.chat_agent:
    st.session_state.chat_agent[session_id] = ChatAgent()
if session_id not in st.session_state.embedded_files:
    st.session_state.embedded_files[session_id] = set()
if session_id not in st.session_state.chat_histories:
    st.session_state.chat_histories[session_id] = []

# Sidebar
st.sidebar.title("Sessões de Chat")


# Campo para criar nova sessão
st.sidebar.markdown("---")
new_session_id = st.sidebar.text_input("Nova sessão ID", value="")
if st.sidebar.button("Criar nova sessão"):
    new_session_id = str(new_session_id).strip()
    if new_session_id and new_session_id not in st.session_state.chat_histories:
        st.session_state.chat_histories[new_session_id] = []
        st.session_state.session_id = new_session_id

# Mostra botões para cada sessão existente
st.sidebar.markdown("# Sessões Ativas:")
for sid in st.session_state.chat_histories.keys():
    sid_str = str(sid)  # garante string para o botão e key
    if st.sidebar.button(f"Sessão {sid_str}", key=f"botao_sessao_{sid_str}"):
        st.session_state.session_id = sid_str


# Exibe histórico atual
current_session_id = st.session_state.session_id
chat_history = st.session_state.chat_histories.get(current_session_id, [])

for msg in chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

#### Input do usuário
prompt = st.chat_input(
    "Digite sua mensagem ou insira um arquivo PDF ou CSV",
    accept_file="multiple",
    file_type=["pdf", "csv"],
)
if prompt and prompt["files"]:
    for uploaded_file in prompt["files"]:
        filename = uploaded_file.name
        ext = filename.split(".")[-1].lower()
        # st.write("DEBUG: extenção arquivo:", ext)
        if filename in st.session_state.embedded_files[session_id]:
            # st.info(f"DEBUG: O arquivo '{filename}' já foi embeddado nesta sessão. Ignorando.")
            continue
        with st.spinner("Vetorizando o arquivo..."):

            loader = CustomLoader(file=uploaded_file, filename=filename)
            docs = loader.load()
            uploaded_file.close()
            if ext == "pdf":
                processor = EmbeddingProcessor(data=docs, session_id=session_id)
                retriever = processor.create_retriever()
                st.session_state.chat_agent[session_id] = ChatAgent(retriever=retriever)
                st.session_state.embedded_files[session_id].add(filename)
                # Adiciona mensagem do sistema ao histórico
                add_system_message(
                    session_id,
                    f"Voce agora possui conhecimento sobre o arquivo PDF (que poser ser textos, artigos e outros) de nome: '{filename}'! \nResponda perguntas sobre o conteúdo desse arquivo. \nSe não souber a resposta, seja honesto e proponha caminhos para buscar a informação. \nSe tiver mais de um arquivo, use o nome do arquivo para referenciar o conteúdo que deseja consultar.",
                )
                st.session_state.chat_histories[session_id].append(
                    {
                        "role": "system",
                        "content": f"Arquivo '{filename}' embeddado com sucesso!",
                    }
                )
                st.success(f"PDF '{filename}' embeddado com sucesso!")
            elif ext == "csv":
                # Criar novo agente com DataFrame
                st.write("DEBUG: docs:", docs)
                # Atualizar agente com novo contexto
                st.session_state.chat_agent[session_id] = ChatAgent(retriever=None)
                st.session_state.chat_agent[session_id].load_dataframe_tools(df=docs)
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

if prompt and prompt.text:
    st.chat_message("user").markdown(prompt.text)

    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            resposta = st.session_state.chat_agent[current_session_id].responder(
                prompt.text, session_id=current_session_id
            )
            ### Debug para mostrar o conteúdo da resposta no streamlit (Historico de mensagens, Contexto (se tiver embeddings), etc.)
            # st.write("DEBUG: tipo de resposta:", type(resposta))
            st.write("DEBUG: conteúdo:", resposta)

            resposta_texto = (
                resposta.get("answer")
                if isinstance(resposta, dict)
                else resposta.content
            )
            st.markdown(resposta_texto)

    # Atualiza histórico dessa sessão
    if current_session_id not in st.session_state.chat_histories:
        st.session_state.chat_histories[current_session_id] = []

    st.session_state.chat_histories[current_session_id].append(
        {"role": "user", "content": prompt.text}
    )
    st.session_state.chat_histories[current_session_id].append(
        {"role": "assistant", "content": resposta_texto}
    )
