FROM python:3.11-slim

WORKDIR /app

# Install Python dependencies before copying source so Docker can cache the layer.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
