from agents_ia.chat import ChatAgent

agente = ChatAgent()

print("Assistente LLaMA 2 (digite 'sair' para encerrar)\n")

while True:
    entrada = input("Você: ")
    if entrada.lower() in ["sair", "exit", "quit"]:
        break
    resposta = agente.responder(entrada, session_id=1)
    print(f"Assistente: ", resposta.content)
    print("\n")
print("Sessão encerrada.")
