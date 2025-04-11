FROM python:3.12.6-slim

WORKDIR /app
COPY . /app

RUN rm -rf /app/chroma_storage

RUN pip install --upgrade pip
RUN pip install -r requirements-docker.txt

ENV OLLAMA_URL=http://host.docker.internal:11434

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]