from agents_ia.chat import ChatAgent

agente = ChatAgent()

print("Assistente LLaMA 2 (digite 'sair' para encerrar)\n")

while True:
    entrada = input("Você: ")
    if entrada.lower() in ["sair", "exit", "quit"]:
        break

    stream = agente.responder(entrada, session_id="foo")
    print(f"Assistente: ")
    for chunk in stream:
        print(chunk, end="", flush=True)
    print("\n")
print("Sessão encerrada.")
