import logging
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score
)

# =====================================================
# ИМПОРТ ГЕНЕРАТОРА ДАННЫХ (исправлено)
# =====================================================

from data_generator.generator import (
    generate_fraud_subset,
    get_expected_columns
)

# =====================================================
# ИМПОРТ STRESS SCENARIOS
# =====================================================

from data_generator.stress_scenarios import (
    apply_stress,
    get_available_scenarios
)

# =====================================================
# ЛОГИРОВАНИЕ
# =====================================================

logging.basicConfig(
    filename="app.log",
    level=logging.ERROR,
    format="%(asctime)s %(levelname)s %(message)s"
)

# =====================================================
# НАСТРОЙКА СТРАНИЦЫ
# =====================================================

st.set_page_config(
    page_title="Система антифрода",
    layout="wide"
)

st.title("Система антифрода")

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.header("Настройки")

# Размер выборки (ограничение от 100 до 2000, как в generate_fraud_subset)
sample_size = st.sidebar.slider(
    "Размер выборки",
    min_value=100,
    max_value=2000,
    value=1000,
    step=100,
    help="Количество генерируемых транзакций (от 100 до 2000)"
)

# Порог классификации
threshold = st.sidebar.slider(
    "Порог классификации",
    min_value=0.1,
    max_value=0.9,
    value=0.5,
    step=0.05,
    help="Вероятность выше этого порога считается мошенничеством"
)

# Доля фрода
fraud_ratio = st.sidebar.slider(
    "Доля мошеннических транзакций",
    min_value=0.01,
    max_value=0.3,
    value=0.05,
    step=0.01,
    format="%.2f",
    help="Процент мошеннических транзакций в данных"
)

# Stress scenario
available_scenarios = get_available_scenarios()

selected_scenario = st.sidebar.selectbox(
    "Стресс-сценарий",
    available_scenarios,
    help="Имитация аномального поведения мошенников"
)

# Upload CSV
uploaded_file = st.file_uploader(
    "Загрузите CSV файл",
    type=["csv"],
    help="Файл должен содержать колонки, соответствующие обучающим данным и is_fraud"
)

# Информация о моделях
st.sidebar.markdown("---")
st.sidebar.markdown("### Доступные модели")
st.sidebar.caption("Logistic Regression (450k)")
st.sidebar.caption("Random Forest v1 (450k)")
st.sidebar.caption("Random Forest v2 (450k)")
st.sidebar.caption("CatBoost v1 (450k)")
st.sidebar.caption("CatBoost v2 (450k)")
st.sidebar.caption("Isolation Forest (450k)")

# Кнопка запуска
compare_button = st.sidebar.button(
    "Запустить анализ",
    type="primary",
    help="Нажмите для расчёта метрик на выбранных данных"
)

# =====================================================
# ЗАГРУЗКА МОДЕЛЕЙ (кэширование)
# =====================================================
@st.cache_resource
def load_all_models():
    """Загружает все 6 моделей, обученных на 450k данных"""
    models = {}
    
    # Модели из папки 450k_models
    models['Logistic Regression'] = joblib.load('models/450k_models/logistic_regression_450k.pkl')
    models['Random Forest v1'] = joblib.load('models/450k_models/random_forest_v1_450k.pkl')
    models['Random Forest v2'] = joblib.load('models/450k_models/random_forest_v2_450k.pkl')
    
    # Модели из папки advanced_models
    models['CatBoost v1'] = joblib.load('models/advanced_models/catboost_v1.pkl')
    models['CatBoost v2'] = joblib.load('models/advanced_models/catboost_v2.pkl')
    models['Isolation Forest'] = joblib.load('models/advanced_models/isolation_forest.pkl')
    
    return models

# =====================================================
# ОСНОВНАЯ ЛОГИКА
# =====================================================

if compare_button:
    st.toast(f"Запуск анализа со сценарием: {selected_scenario}")

    try:
        # =================================================
        # ЗАГРУЗКА CSV ИЛИ ГЕНЕРАЦИЯ ДАННЫХ
        # =================================================
        if uploaded_file is not None:
            if not uploaded_file.name.endswith(".csv"):
                st.error("Некорректное расширение файла")
                st.stop()
            classic_df = pd.read_csv(uploaded_file)
            st.success(f"Загружен файл: {uploaded_file.name}")
        else:
            with st.spinner("Генерируем синтетические данные, пожалуйста, подождите..."):
                # Используем generate_fraud_subset вместо generate_transactions
                classic_df = generate_fraud_subset(
                    subset_size=sample_size,
                    full_size=2000,
                    fraud_ratio=fraud_ratio,
                    label_noise=0.015,
                    random_state=42,
                    use_stratification=True
                )
                # generate_fraud_subset возвращает DataFrame с колонкой is_fraud

        # =================================================
        # СТРЕСС-СЦЕНАРИЙ
        # =================================================
        stress_df = apply_stress(classic_df.copy(), selected_scenario)

        # =================================================
        # ОТОБРАЖЕНИЕ DATASET
        # =================================================
        st.subheader("Данные")
        st.dataframe(classic_df.head(), use_container_width=True)

        # =================================================
        # ВАЛИДАЦИЯ КОЛОНОК
        # =================================================
        required_columns = get_expected_columns() + ['is_fraud']
        missing_columns = [col for col in required_columns if col not in classic_df.columns]

        if missing_columns:
            st.error(f"Отсутствуют колонки: {missing_columns}")
            st.stop()

        # =================================================
        # ВАЛИДАЦИЯ ТИПОВ
        # =================================================
        wrong_types = []
        for col in get_expected_columns():
            if col in classic_df.columns and not pd.api.types.is_numeric_dtype(classic_df[col]):
                wrong_types.append(col)

        if wrong_types:
            st.error(f"Неверный тип данных: {wrong_types}")
            st.stop()

        # =================================================
        # ИНТЕРАКТИВНОЕ РЕДАКТИРОВАНИЕ
        # =================================================
        st.subheader("Редактирование датасета")
        edited_df = st.data_editor(classic_df, num_rows="dynamic")

        # =================================================
        # FEATURES / TARGET
        # =================================================
        feature_cols = get_expected_columns()
        X_classic = edited_df[feature_cols]
        y_classic = edited_df["is_fraud"]
        X_stress = stress_df[feature_cols]
        y_stress = stress_df["is_fraud"]

        # =================================================
        # ЗАГРУЗКА ВСЕХ 6 МОДЕЛЕЙ
        # =================================================
        models = load_all_models()
        st.success(f"Загружено {len(models)} моделей")

        # =================================================
        # РАСЧЁТ МЕТРИК ДЛЯ ВСЕХ МОДЕЛЕЙ
        # =================================================
        results_classic = []
        results_stress = []

        for name, model in models.items():
            # Classic predictions
            if hasattr(model, "predict_proba"):
                classic_proba = model.predict_proba(X_classic)[:, 1]
                classic_pred = (classic_proba >= threshold).astype(int)
            else:
                classic_pred = model.predict(X_classic)
            
            # Stress predictions
            if hasattr(model, "predict_proba"):
                stress_proba = model.predict_proba(X_stress)[:, 1]
                stress_pred = (stress_proba >= threshold).astype(int)
            else:
                stress_pred = model.predict(X_stress)
            
            # Метрики classic
            results_classic.append({
                "Модель": name,
                "Precision": round(precision_score(y_classic, classic_pred, zero_division=0), 4),
                "Recall": round(recall_score(y_classic, classic_pred, zero_division=0), 4),
                "F1": round(f1_score(y_classic, classic_pred, zero_division=0), 4)
            })
            
            # Метрики stress
            results_stress.append({
                "Модель": name,
                "Precision": round(precision_score(y_stress, stress_pred, zero_division=0), 4),
                "Recall": round(recall_score(y_stress, stress_pred, zero_division=0), 4),
                "F1": round(f1_score(y_stress, stress_pred, zero_division=0), 4)
            })

        # =================================================
        # ТАБЛИЦЫ МЕТРИК
        # =================================================
        df_classic = pd.DataFrame(results_classic)
        df_stress = pd.DataFrame(results_stress)

        # Форматирование в проценты
        for col in ['Precision', 'Recall', 'F1']:
            df_classic[col] = df_classic[col].apply(lambda x: f"{x*100:.1f}%")
            df_stress[col] = df_stress[col].apply(lambda x: f"{x*100:.1f}%")

        # Таблица 1: Классический режим
        st.subheader("Таблица метрик (классический режим)")
        st.dataframe(df_classic, use_container_width=True, hide_index=True)

        # Таблица 2: Стресс-режим
        st.subheader(f"Таблица метрик (стресс-режим: {selected_scenario})")
        st.dataframe(df_stress, use_container_width=True, hide_index=True)

        # =================================================
        # ПРОСАДКА МЕТРИК
        # =================================================
        st.subheader("Просадка метрик при стрессе (%)")

        drop_data = []
        for i, row in df_classic.iterrows():
            classic_f1 = float(row['F1'].replace('%', ''))
            stress_f1 = float(df_stress.iloc[i]['F1'].replace('%', ''))
            drop_f1 = round(((classic_f1 - stress_f1) / classic_f1) * 100, 1) if classic_f1 > 0 else 0
            
            drop_data.append({
                "Модель": row['Модель'],
                "Просадка Precision": "-",
                "Просадка Recall": "-",
                "Просадка F1": f"{drop_f1}%"
            })

        df_drop = pd.DataFrame(drop_data)
        st.dataframe(df_drop, use_container_width=True, hide_index=True)

        # =================================================
        # УСПЕШНОЕ ЗАВЕРШЕНИЕ
        # =================================================
        st.success("Анализ завершён")

    except FileNotFoundError as e:
        logging.error(f"Файл модели не найден: {e}")
        st.error(f"Файл модели не найден: {e}")
        st.info("Убедитесь, что модели сохранены в папках `models/450k_models/` и `models/advanced_models/`")

    except Exception as e:
        logging.error(str(e))
        st.error(f"Ошибка приложения: {e}")