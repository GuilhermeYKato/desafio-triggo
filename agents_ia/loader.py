from langchain_community.document_loaders import UnstructuredPDFLoader, CSVLoader
from pathlib import Path
import tempfile


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
            return loader.load()

    def _load_csv(self):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            tmp.write(self.file.read())
            tmp.flush()
            loader = CSVLoader(tmp.name)
            return loader.load()
