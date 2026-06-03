FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Копируем только необходимые части проекта
COPY app.py .

COPY data_generator ./data_generator
COPY metrics ./metrics
COPY database ./database
COPY models ./models

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]