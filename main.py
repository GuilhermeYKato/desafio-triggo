import streamlit as st
from agents_ia.chat import ChatAgent

st.set_page_config(page_title="Chat com LLaMA3", page_icon="🐉")
st.title("💬 Chatbot com LLaMA3")

# Simular múltiplos chats
if "session_id" not in st.session_state:
    st.session_state.session_id = 1

if "chat_agent" not in st.session_state:
    st.session_state.chat_agent = ChatAgent()

if "chat_histories" not in st.session_state:
    st.session_state.chat_histories = {}


# Sidebar para trocar de chat/sessão
st.sidebar.title("Sessões de Chat")

# Garante que o session_id atual é string
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
    st.chat_message("assistant").markdown(resposta.content)

    # Atualiza histórico dessa sessão
    if current_session_id not in st.session_state.chat_histories:
        st.session_state.chat_histories[current_session_id] = []

    st.session_state.chat_histories[current_session_id].append(
        {"role": "user", "content": prompt}
    )
    st.session_state.chat_histories[current_session_id].append(
        {"role": "assistant", "content": resposta.content}
    )
