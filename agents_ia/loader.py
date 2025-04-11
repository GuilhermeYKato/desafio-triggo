from langchain_community.document_loaders import UnstructuredPDFLoader, CSVLoader
from pathlib import Path
import tempfile
import pandas as pd


class CustomLoader:
    def __init__(self, file, filename: str):
        """
        Inicializa o carregador personalizado com um arquivo enviado via upload.

        file: objeto do tipo BytesIO (retornado por st.file_uploader)
        filename: string representando o nome do arquivo, necessário para saber a extensão (.pdf ou .csv)

        return: None
        """
        self.file = file
        self.filename = filename

    def _load(self):
        """
        Carrega o conteúdo do arquivo com base na sua extensão (.pdf ou .csv).

        return:
            - list de Document (se for PDF)
            - pandas.DataFrame (se for CSV)

        raise:
            - ValueError se a extensão do arquivo não for suportada.
        """
        ext = Path(self.filename).suffix.lower()
        if ext == ".pdf":
            return self._load_pdf()
        elif ext == ".csv":
            return self._load_csv()
        else:
            raise ValueError(f"Formato de arquivo não suportado: {ext}")

    def _load_pdf(self):
        """
        Carrega e processa um arquivo PDF utilizando o UnstructuredPDFLoader.
        Adiciona metadados personalizados ao documento indicando o nome do arquivo original.

        return:
            - list de Document: documentos extraídos do PDF com metadados incluídos.

        raise:
            - Exception se houver erro durante a leitura ou parsing do PDF.
        """
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(self.file.read())
            tmp.flush()
            temp_path = tmp.name

            try:
                loader = UnstructuredPDFLoader(temp_path)
                docs = loader.load()
                for doc in docs:
                    doc.metadata["Arquivo"] = self.filename
                return docs

            except Exception as e:
                print(f"Erro ao carregar o PDF: {e}")
                raise e

    def _load_csv(self):
        """
        Carrega e processa um arquivo CSV utilizando o pandas.

        return:
            - pandas.DataFrame: dataframe contendo os dados do arquivo CSV.

        raise:
            - Exception se houver erro durante a leitura do CSV.
        """
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            tmp.write(self.file.read())
            tmp.flush()
            temp_path = tmp.name

            try:
                df = pd.read_csv(temp_path)
                return df

            except Exception as e:
                print(f"Erro ao carregar o CSV: {e}")
                raise e
