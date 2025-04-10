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
        self.data = data
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.session_id = str(session_id)
        self.embeddings = EmbeddingLLM().embedding_llm
        self.persist_path = f"./chroma_storage/session_{self.session_id}"
        self.collection_name = f"session_{self.session_id}"

    def create_retriever(self):

        docs = self.data
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )
        splits = splitter.split_documents(docs)
        print(f"Total de splits: {len(splits)}")

        # Verifica se já existe diretório com dados persistidos
        if os.path.exists(self.persist_path):
            # Abre o vetorstore existente e adiciona novos documentos
            vectorstore = Chroma(
                embedding_function=self.embeddings,
                persist_directory=self.persist_path,
                collection_name=self.collection_name,
            )
            vectorstore.add_documents(splits)
        else:

            vectorstore = Chroma.from_documents(
                documents=splits,
                embedding=self.embeddings,
                collection_name=self.collection_name,
                persist_directory=self.persist_path,
            )

        return vectorstore.as_retriever(search_kwargs={"k": 8})
