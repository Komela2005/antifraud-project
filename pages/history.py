import streamlit as st
import pandas as pd
import sqlite3

st.set_page_config(page_title="Experiment History", layout="wide")
st.title("Experiment History")

# Подключение к БД
conn = sqlite3.connect('experiments.db')

# Загрузка данных
try:
    experiments = pd.read_sql_query("SELECT * FROM experiments ORDER BY timestamp DESC", conn)
    
    if experiments.empty:
        st.info("No experiments found. Run analysis first.")
    else:
        # Фильтры
        st.sidebar.header("Filters")
        
        # Фильтр по модели
        conn2 = sqlite3.connect('experiments.db')
        models_df = pd.read_sql_query("SELECT DISTINCT model_name FROM model_results", conn2)
        models = ["All"] + models_df['model_name'].tolist()
        selected_model = st.sidebar.selectbox("Filter by model", models)
        
        # Фильтр по режиму
        modes = ["All", "classic", "stress"]
        selected_mode = st.sidebar.selectbox("Filter by mode", modes)
        
        # Фильтр по дате
        date_range = st.sidebar.date_input("Date range", [])
        
        # Загрузка деталей
        results = pd.read_sql_query("""
            SELECT e.id, e.timestamp, e.sample_size, e.threshold, e.fraud_ratio, 
                   e.stress_scenario, mr.model_name, mr.mode, mr.precision, mr.recall, mr.f1, mr.business_cost
            FROM experiments e
            JOIN model_results mr ON e.id = mr.experiment_id
            ORDER BY e.timestamp DESC
        """, conn)
        
        # Применение фильтров
        if selected_model != "All":
            results = results[results['model_name'] == selected_model]
        if selected_mode != "All":
            results = results[results['mode'] == selected_mode]
        
        # Форматирование
        results['precision'] = (results['precision'] * 100).round(1).astype(str) + '%'
        results['recall'] = (results['recall'] * 100).round(1).astype(str) + '%'
        results['f1'] = (results['f1'] * 100).round(1).astype(str) + '%'
        
        st.dataframe(results, use_container_width=True, hide_index=True)
        
        # Экспорт
        csv = results.to_csv(index=False).encode('utf-8')
        st.download_button("Download as CSV", csv, "experiments.csv", "text/csv")
        
except Exception as e:
    st.error(f"Error loading history: {e}")
finally:
    conn.close()
