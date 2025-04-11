import subprocess
import shutil
import os


def main():
    # Caminho para a pasta chroma_storage
    chroma_storage_path = os.path.join(os.getcwd(), "chroma_storage")
    print(f"Verificando a pasta '{chroma_storage_path}'...")
    # Excluir a pasta chroma_storage, se existir
    if os.path.exists(chroma_storage_path):
        shutil.rmtree(chroma_storage_path)
        print(f"Pasta '{chroma_storage_path}' excluída com sucesso.")
    else:
        print(f"Pasta '{chroma_storage_path}' não encontrada.")

    # Rodar o script streamlit
    command = "streamlit run app.py"
    subprocess.run(command, shell=True)


if __name__ == "__main__":
    main()
