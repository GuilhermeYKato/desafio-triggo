import streamlit as st
from agents_ia.chat import ChatAgent
from agents_ia.embedding import EmbeddingProcessor
from agents_ia.loader import CustomLoader

st.set_page_config(page_title="Chat com LLaMA3", page_icon="🐉")
st.title("Chatbot com LLaMA3")

# Estado inicial
if "session_id" not in st.session_state:
    st.session_state.session_id = "1"

if "chat_agent" not in st.session_state:
    st.session_state.chat_agent = ChatAgent()

if "chat_histories" not in st.session_state:
    st.session_state.chat_histories = {}

if "last_uploaded_filename" not in st.session_state:
    st.session_state.last_uploaded_filename = None

# Sidebar
st.sidebar.title("Sessões de Chat")

# Upload de arquivos no topo
uploaded_file = st.sidebar.file_uploader("Envie um PDF ou CSV", type=["pdf", "csv"])
if uploaded_file:
    filename = uploaded_file.name
    ext = filename.split(".")[-1].lower()
    st.write("DEBUG: extenção arquivo:", ext)
    if filename != st.session_state.last_uploaded_filename:
        st.session_state.last_uploaded_filename = filename

        with st.sidebar:
            with st.spinner("Vetorizando o arquivo..."):
                if ext == "pdf":
                    # Criar novo retriever
                    loader = CustomLoader(file=uploaded_file, filename=filename)
                    docs = loader.load()
                    uploaded_file.close()
                    processor = EmbeddingProcessor(
                        data=docs,
                        session_id=st.session_state.session_id,
                    )
                    # Precisa checar se é true para a sessão atual
                    retriever = processor.create_retriever()

                    # Atualizar agente com novo contexto
                    st.session_state.chat_agent = ChatAgent(retriever=retriever)
                    retriever = None
                elif ext == "csv":
                    # Criar novo agente com DataFrame
                    loader = CustomLoader(file=uploaded_file, filename=filename)
                    docs = loader.load()
                    st.write("DEBUG: docs:", docs)
                    # Atualizar agente com novo contexto
                    st.session_state.chat_agent = ChatAgent(retriever=None)
                    st.session_state.chat_agent.load_dataframe_tools(df=docs)
            st.success(f"Arquivo '{filename}' processado com sucesso!")

# Controle de sessões
st.session_state.session_id = str(st.session_state.session_id)
current_session_id = st.session_state.session_id

# Inicializa histórico se necessário
if current_session_id not in st.session_state.chat_histories:
    st.session_state.chat_histories[current_session_id] = []


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


# Input do usuário
prompt = st.chat_input("Digite sua mensagem...")
if prompt:
    st.chat_message("user").markdown(prompt)

    resposta = st.session_state.chat_agent.responder(
        prompt, session_id=current_session_id
    )

    # st.write("DEBUG: tipo de resposta:", type(resposta))
    st.write("DEBUG: conteúdo:", resposta)

    resposta_texto = (
        resposta.get("answer") if isinstance(resposta, dict) else resposta.content
    )
    st.chat_message("assistant").markdown(resposta_texto)
    # Atualiza histórico dessa sessão
    if current_session_id not in st.session_state.chat_histories:
        st.session_state.chat_histories[current_session_id] = []

    st.session_state.chat_histories[current_session_id].append(
        {"role": "user", "content": prompt}
    )
    st.session_state.chat_histories[current_session_id].append(
        {"role": "assistant", "content": resposta_texto}
    )
