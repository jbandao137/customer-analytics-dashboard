FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY data/ data/
COPY models/ models/
COPY src/ src/

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "src/dashboard.py", \
            "--server.port=8501", \
            "--server.address=0.0.0.0"]
