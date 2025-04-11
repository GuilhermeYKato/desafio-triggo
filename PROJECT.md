# Configurando o Ambiente

## Instalação Ollama

Para instalar o **Ollama**, acesse o site oficial:
[https://ollama.com/download](https://ollama.com/download)

Você pode consultar todos os modelos disponíveis em:
[https://ollama.com/search](https://ollama.com/search)

### Baixando os modelos

Após instalar o Ollama, abra o terminal e execute os comandos:
- **Modelo LLM (Llama 3.2):**
```bash
ollama pull llama3.2
```

- **Modelo de Embedding (mxbai-embed-large):**
```bash
ollama pull mxbai-embed-large
```

Se o Ollama estiver instalado corretamente, o download dos modelos será iniciado automaticamente após cada comando.

# Configuração do Python
## Visual Studio Code (VS Code)
1. Instalar a extensão **Python** no VS Code.
2. Abrir a barra de comandos do VS Code (Comando padrão: `Ctrl + Shift + P`). 
3. Digitar `Python: Create Environment` para criar um **ambiente virtual**.
4. Após criado, o VS Code deve ativar o ambiente automaticamente. Se não ativar, veja abaixo como fazer via terminal.
5. Instale as dependências do projeto:
```bash
pip install -r requirements.txt
```


## Terminal
1. Crie o ambiente virtual para python:
```bash
python -m venv .venv
```

2. Ative o ambiente virtual:
    - **Windows:**
    ```bash
    ./.venv/Scripts/activate
    ```
    - **Linux/macOS:**
    ```bash
    source .venv/bin/activate
    ```

3. Instale os pacotes listados no arquivo requirements.txt:
```bash
pip install -r requirements.txt
```

4. Para desativar o ambiente virtual:
```bash
deactivate
```

# Compilar o projeto

Garanta que o Ollama esta sendo executado na sua máquina.

Com todas as dependências instaladas e o Ollama em execução, compile e execute o arquivo `main.py` utilizando o comando abaixo:

```bash
python main.py
```

Se tudo estiver configurado corretamente, uma página com o chat do Streamlit será aberta no seu navegador.

# Projeto

## Geral

O projeto implementa um chatbot iterativo (Assistente Virtual) com memória de sessões, em que é capaz de contextualização baseada em documentos enviados pelos usuários (PDF e CSV).
Foi desenvolvido com a biblioteca **Langchain** explorando recursos como memória conversacional, ferramentas personalizadas e prompt pré-feitos. A interface foi feita com **Streamlit**. 
O LLM local escolhido e utilizado foi o **LLaMA3**, especificamente o **LLaMA3.2** de 3 bilhões de parametros. Para a criação dos embeddings foi escolhido **mxbai-embed-large** com 335 milhões de parametros. Todos eles podem ser acessados localmente via `Ollama`.

## Funcionalidades

1. **Múltiplas Sessões de Chat**: Cada conversa é isolada por ID de sessão, mantendo histórico e arquivos embeddados próprios.
2. **Memória de Conversa**: Usa `InMemoryChatMessageHistory` para reter mensagens trocadas, incluindo `SystemMessage` com instruções comportamentais.
3. **Contextualização via RAG (Retrieval-Augmented Generation)**: Arquivos PDF são divididos em trechos com o `RecursiveCharacterTextSplitter`, embeddados e armazenados em um vetorstore (`Chroma`) para fornecer contexto relevante às respostas.
4. **Criação de Embeddings**: Os chunks extraídos dos PDFs são vetorizados com o modelo `mxbai-embed-large` e persistidos em um banco local `Chroma` por sessão.
5. **Suporte a CSVs**: Arquivos CSV são carregados como DataFrame e o agente ganha ferramentas (`PythonAstREPLTool`) para análise com `pandas` e consegue executar comandos basicos.
6. **Atualização Dinâmica**: O agente troca automaticamente entre modos (`llm, rag, csv`) conforme os arquivos são carregados.
7. **Interface com Streamlit**: Usuários podem conversar com o assistente e enviar múltiplos arquivos por sessão, com feedback visual do status do carregamento e embeddings.

## Estrutura do Código

1. memory.py
    - Define e gerencia o histórico de `mensagens por sessão`.
    - Ao iniciar uma nova sessão, adiciona uma `SystemMessage` com instruções comportamentais padronizadas.
    - Função `add_system_message()` permite adicionar instruções contextuais adicionais após carregamento de arquivos.

2. chat.py
    - Define o `ChatAgent`, um wrapper que gerencia:
    - Modo de resposta padrão (`llm`)
    - RAG com retriever (`rag`)
    - Análise de dados (`csv`)
    - Possui método `responder()` para responder com base no modo atual.

3. loader.py
    - Contém `CustomLoader`, que processa arquivos PDF ou CSV.
    - PDF é processado em Document com texto dividido para embeddings.
    - CSV retorna um `pandas.DataFrame`.

4. embedding.py
    - Cria embeddings com o `EmbeddingProcessor` que são salvo com `Chroma` e inicializa um `retriever` por sessão.
    - Cada retriever é separado por `session_id`.

5. local_llm.py
    - Inicia o modelo local `LLaMa3.2` com `ChatOllama`.
    - Inicia o modelo local `mxbai-embed-large` com `OllamaEmbeddings`.

6. app.py 
    - Design de toda aplicação Frontend com `Streamlit`.
    - Gerenciamento dos chats por `session_id`. Cada chat tem sua memória propria.
    - Realização de upload dos arquivos.
    - Exibição do histórico de mensagens e respostas do assistente.

7. main.py
    - Função de excluir dados antigos dos `Chroma`.
    - Automaticar a execução do `Streamlit`.