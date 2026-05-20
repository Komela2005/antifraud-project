# =====================================================
# ИМПОРТ БИБЛИОТЕК
# =====================================================

import logging
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# =====================================================
# ИМПОРТ ГЕНЕРАТОРА ДАННЫХ
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
# ФУНКЦИЯ РАСЧЁТА БИЗНЕС-СТОИМОСТИ
# =====================================================

def calculate_business_cost(y_true, y_pred, fp_weight, fn_weight):
    """Рассчитывает бизнес-стоимость ошибок классификации"""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    total_cost = fp * fp_weight + fn * fn_weight
    return total_cost

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

# Размер выборки
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

# Cost matrix веса
st.sidebar.markdown("---")
st.sidebar.subheader("Стоимость ошибок (Cost Matrix)")

fp_weight = st.sidebar.slider(
    "Стоимость False Positive (ложная тревога)",
    min_value=1,
    max_value=100,
    value=1,
    step=1,
    help="Штраф за блокировку обычного клиента"
)

fn_weight = st.sidebar.slider(
    "Стоимость False Negative (пропущенный фрод)",
    min_value=1,
    max_value=100,
    value=10,
    step=1,
    help="Штраф за пропуск мошеннической транзакции"
)

# Stress scenario
available_scenarios = get_available_scenarios()

selected_scenario = st.sidebar.selectbox(
    "Стресс-сценарий",
    available_scenarios,
    help="Имитация аномального поведения мошенников"
)

# =====================================================
# ВЫБОР МОДЕЛЕЙ (мультиселект)
# =====================================================

st.sidebar.markdown("---")
st.sidebar.subheader("Выбор моделей для сравнения")

all_models = [
    "Logistic Regression",
    "Random Forest v1",
    "Random Forest v2",
    "CatBoost v1",
    "CatBoost v2",
    "Isolation Forest"
]

selected_models = st.sidebar.multiselect(
    "Модели для сравнения (можно выбрать несколько)",
    options=all_models,
    default=all_models,
    help="Выберите одну или несколько моделей для сравнения"
)

# Upload CSV (опционально, если не загружен - используем генерацию)
uploaded_file = st.file_uploader(
    "Загрузите CSV файл (опционально)",
    type=["csv"],
    help="Если не загружать, будут сгенерированы синтетические данные"
)

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
    
    models['Logistic Regression'] = joblib.load('models/450k_models/logistic_regression_450k.pkl')
    models['Random Forest v1'] = joblib.load('models/450k_models/random_forest_v1_450k.pkl')
    models['Random Forest v2'] = joblib.load('models/450k_models/random_forest_v2_450k.pkl')
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
            
            # ВАЛИДАЦИЯ КОЛОНОК ДЛЯ ЗАГРУЖЕННОГО ФАЙЛА
            required_columns = get_expected_columns() + ['is_fraud']
            missing_columns = [col for col in required_columns if col not in classic_df.columns]
            
            if missing_columns:
                st.error(f"Отсутствуют колонки в загруженном файле: {missing_columns}")
                st.info("Используйте синтетическую генерацию или дополните CSV недостающими колонками")
                st.stop()
        else:
            with st.spinner("Генерируем синтетические данные..."):
                classic_df = generate_fraud_subset(
                    subset_size=sample_size,
                    full_size=2000,
                    fraud_ratio=fraud_ratio,
                    label_noise=0.015,
                    random_state=42,
                    use_stratification=True
                )

        # =================================================
        # СТРЕСС-СЦЕНАРИЙ
        # =================================================
        stress_df = apply_stress(classic_df.copy(), selected_scenario)

        # =================================================
        # ОТОБРАЖЕНИЕ DATASET
        # =================================================
        st.subheader("Сгенерированные данные")
        st.dataframe(classic_df.head(), use_container_width=True)

        # =================================================
        # ВАЛИДАЦИЯ ТИПОВ
        # =================================================
        feature_cols = get_expected_columns()
        wrong_types = []
        for col in feature_cols:
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
        X_classic = edited_df[feature_cols]
        y_classic = edited_df["is_fraud"]
        X_stress = stress_df[feature_cols]
        y_stress = stress_df["is_fraud"]

        # =================================================
        # ЗАГРУЗКА ВСЕХ 6 МОДЕЛЕЙ
        # =================================================
        all_models_dict = load_all_models()
        
        # Фильтруем только выбранные модели
        models = {name: all_models_dict[name] for name in selected_models if name in all_models_dict}
        
        st.success(f"Загружено {len(models)} моделей: {', '.join(models.keys())}")

        # =================================================
        # РАСЧЁТ МЕТРИК ДЛЯ ВЫБРАННЫХ МОДЕЛЕЙ
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
            
            # Бизнес-стоимость
            business_cost_classic = calculate_business_cost(y_classic, classic_pred, fp_weight, fn_weight)
            business_cost_stress = calculate_business_cost(y_stress, stress_pred, fp_weight, fn_weight)
            
            # Метрики classic
            results_classic.append({
                "Модель": name,
                "Precision": round(precision_score(y_classic, classic_pred, zero_division=0), 4),
                "Recall": round(recall_score(y_classic, classic_pred, zero_division=0), 4),
                "F1": round(f1_score(y_classic, classic_pred, zero_division=0), 4),
                "Business Cost": business_cost_classic
            })
            
            # Метрики stress
            results_stress.append({
                "Модель": name,
                "Precision": round(precision_score(y_stress, stress_pred, zero_division=0), 4),
                "Recall": round(recall_score(y_stress, stress_pred, zero_division=0), 4),
                "F1": round(f1_score(y_stress, stress_pred, zero_division=0), 4),
                "Business Cost": business_cost_stress
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
        
        # Business Cost оставляем как число
        df_classic['Business Cost'] = df_classic['Business Cost'].astype(int)
        df_stress['Business Cost'] = df_stress['Business Cost'].astype(int)

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
                "Просадка F1": f"{drop_f1}%",
                "Classic Business Cost": df_classic.iloc[i]['Business Cost'],
                "Stress Business Cost": df_stress.iloc[i]['Business Cost']
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