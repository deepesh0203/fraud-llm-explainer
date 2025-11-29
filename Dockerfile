FROM python:3.10

WORKDIR /app

# Install dependencies
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire project
COPY . .

# Make root folder importable (Fix for "No module named utils")
ENV PYTHONPATH="/app"

# Expose Streamlit (8501) and FastAPI (8000)
EXPOSE 8000
EXPOSE 8501

# Start both backend + UI
CMD ["bash", "-c", "export PYTHONPATH=/app && uvicorn backend.main:app --host 0.0.0.0 --port 8000 & streamlit run ui/app.py --server.port=8501 --server.address=0.0.0.0"]
