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

Com todas as dependências instaladas e Ollama rodando. Execute o seguinte comando na raiz do projeto:

```bash
streamlit run main.py
```
