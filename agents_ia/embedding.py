from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from LLM.local_llm import EmbeddingLLM
import os


class EmbeddingProcessor:
    def __init__(
        self,
        data,
        session_id: str,
        chunk_size: int = 1400,
        chunk_overlap: int = 200,
    ):
        """
        Inicializa o processador de embeddings para uma sessão específica.

        data: lista de documentos (Document) a serem processados.
        session_id: identificador único da sessão, usado para nomear diretórios e coleções.
        chunk_size: tamanho de cada chunk (em caracteres) após a divisão dos documentos.
        chunk_overlap: número de caracteres sobrepostos entre chunks consecutivos.

        return: None
        """
        self.data = data
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.session_id = str(session_id)
        self.embeddings = EmbeddingLLM().embedding_llm
        self.persist_path = f"./chroma_storage/session_{self.session_id}"
        self.collection_name = f"session_{self.session_id}"

    def create_retriever(self):
        """
        Cria um retriever baseado em embeddings usando Chroma como vetorstore.
        Os documentos são divididos em chunks antes da indexação.
        Se já existir uma base persistida, ela é carregada e atualizada com os novos documentos.

        return:
            - retriever (VectorStoreRetriever): objeto capaz de recuperar documentos relevantes com base em similaridade vetorial.
        """
        docs = self.data
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )
        splits = splitter.split_documents(docs)
        print(f"Total de splits: {len(splits)}")

        if os.path.exists(self.persist_path):
            # Carrega vetorstore existente e adiciona os novos documentos
            vectorstore = Chroma(
                embedding_function=self.embeddings,
                persist_directory=self.persist_path,
                collection_name=self.collection_name,
            )
            vectorstore.add_documents(splits)
        else:
            # Cria novo vetorstore a partir dos documentos
            vectorstore = Chroma.from_documents(
                documents=splits,
                embedding=self.embeddings,
                collection_name=self.collection_name,
                persist_directory=self.persist_path,
            )

        return vectorstore.as_retriever(search_kwargs={"k": 4})
