from langchain_community.document_loaders import UnstructuredPDFLoader, CSVLoader
from pathlib import Path
import tempfile
import pandas as pd


class CustomLoader:
    def __init__(self, file, filename: str):
        """
        file: objeto do tipo BytesIO (retornado por st.file_uploader)
        filename: necessário para saber a extensão (.pdf ou .csv)
        """
        self.file = file
        self.filename = filename

    def load(self):
        ext = Path(self.filename).suffix.lower()
        if ext == ".pdf":
            return self._load_pdf()
        elif ext == ".csv":
            return self._load_csv()
        else:
            raise ValueError(f"Formato de arquivo não suportado: {ext}")

    def _load_pdf(self):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(self.file.read())
            tmp.flush()
            loader = UnstructuredPDFLoader(tmp.name)
            docs = loader.load()
            temp_path = tmp.name

            try:
                loader = UnstructuredPDFLoader(temp_path)
                docs = loader.load()
                for doc in docs:
                    doc.metadata["Arquivo"] = self.filename
                return docs

            # Handle any exceptions that occur during loading
            except Exception as e:
                print(f"Erro ao carregar o PDF: {e}")
                raise e

    def _load_csv(self):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            tmp.write(self.file.read())
            tmp.flush()
            temp_path = tmp.name

            try:
                df = pd.read_csv(temp_path)
                return df

            # Handle any exceptions that occur during loading
            except Exception as e:
                print(f"Erro ao carregar o CSV: {e}")
                raise e
