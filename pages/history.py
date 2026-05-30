import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st

# Путь к БД
DB_PATH = Path(__file__).parent.parent / "database" / "experiments.db"

st.set_page_config(page_title="Experiment History", layout="wide")
st.title("📊 Experiment History")

# Инициализация БД
try:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from database.db_manager import init_db
    init_db()
except Exception as e:
    st.error(f"❌ Ошибка инициализации БД: {e}")
    st.stop()

# Загрузка и отображение данных
try:
    conn = sqlite3.connect(DB_PATH)
    
    # Проверяем, есть ли эксперименты (исправлено: experiment_date)
    experiments = pd.read_sql_query(
        "SELECT * FROM experiments ORDER BY experiment_date DESC", conn
    )
    
    if experiments.empty:
        st.info("📭 Нет экспериментов. Сначала запустите анализ.")
    else:
        st.sidebar.header("🔍 Фильтры")
        
        # Список моделей
        models_df = pd.read_sql_query(
            "SELECT DISTINCT model_name FROM model_results", conn
        )
        models = ["Все"] + models_df["model_name"].tolist()
        selected_model = st.sidebar.selectbox("Модель", models)
        
        # Режим
        modes = ["Все", "classic", "stress"]
        selected_mode = st.sidebar.selectbox("Режим", modes)
        
        # Результаты (исправлено: experiment_date, f1_score)
        results = pd.read_sql_query(
            """
            SELECT 
                e.id, 
                e.experiment_date as дата,
                e.description as описание,
                mr.model_name as модель, 
                mr.scenario as режим, 
                mr.precision, 
                mr.recall, 
                mr.f1_score as f1, 
                mr.business_cost as стоимость
            FROM experiments e
            JOIN model_results mr ON e.id = mr.experiment_id
            ORDER BY e.experiment_date DESC
        """,
            conn,
        )
        
        # Фильтрация
        if selected_model != "Все":
            results = results[results["модель"] == selected_model]
        if selected_mode != "Все":
            results = results[results["режим"] == selected_mode]
        
        # Форматирование
        if not results.empty:
            results["precision"] = (results["precision"] * 100).round(1).astype(str) + "%"
            results["recall"] = (results["recall"] * 100).round(1).astype(str) + "%"
            results["f1"] = (results["f1"] * 100).round(1).astype(str) + "%"
        
        st.dataframe(results, use_container_width=True, hide_index=True)
        
        # Экспорт
        csv = results.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Скачать CSV", csv, "history.csv", "text/csv")
    
except Exception as e:
    st.error(f"❌ Ошибка загрузки истории: {e}")
    
finally:
    if 'conn' in locals():
        conn.close()