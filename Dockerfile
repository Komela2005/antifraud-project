FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

# Используем облегчённый образ Python 3.10.
# Версия совпадает с настройками Streamlit Cloud.

FROM python:3.10-slim

# Все файлы проекта внутри контейнера будут находиться в /app

WORKDIR /app

# Сначала копируем requirements.txt,
# чтобы Docker мог кэшировать зависимости.

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Копируем весь проект:
# app.py, models, database, data_generator и т.д.

COPY . .

RUN mkdir -p /app/database

EXPOSE 8501

CMD [
    "streamlit",
    "run",
    "app.py",
    "--server.port=8501",
    "--server.address=0.0.0.0"
]