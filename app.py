import logging

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
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return fp * fp_weight + fn * fn_weight

# =====================================================
# НАСТРОЙКА СТРАНИЦЫ
# =====================================================

st.set_page_config(page_title="Система антифрода", layout="wide")
st.title("Система антифрода")

with st.expander("Информация о стресс-сценариях"):
    st.markdown("""
    | Сценарий | Что делает | Зачем нужен |
    |----------|------------|-------------|
    | `normal` | Без изменений | Базовое сравнение (как модель работает в обычном режиме) |
    | `imbalance` | Доля фрода снижается до 0.1% | Проверить, как модель работает, когда мошеннических транзакций очень мало |
    | `amount_shift` | Суммы мошенников уменьшаются в 10 раз | Имитация ситуации, когда мошенники скрывают крупные суммы |
    | `masking` | Признаки мошенников смещаются к норме | Мошенники пытаются выглядеть как обычные пользователи |
    | `frequency_boost` | Все признаки увеличиваются в 1.5 раза | Имитация всплеска активности (массовые атаки) |
    """)

# =====================================================
# SIDEBAR (все виджеты с help)
# =====================================================

st.sidebar.header("Настройки")

sample_size = st.sidebar.slider(
    "Размер выборки", 100, 2000, 1000, 100,
    help="Количество транзакций для анализа"
)

threshold = st.sidebar.slider(
    "Порог классификации", 0.1, 0.9, 0.5, 0.05,
    help="Вероятность выше этого порога считается мошенничеством"
)

fraud_ratio = st.sidebar.slider(
    "Доля мошеннических транзакций", 0.01, 0.30, 0.05, 0.01, format="%.2f",
    help="Доля фрода в синтетических данных"
)

st.sidebar.markdown("---")
st.sidebar.subheader("Стоимость ошибок")

fp_weight = st.sidebar.slider(
    "False Positive", 1, 100, 1,
    help="Стоимость ложной тревоги (блокировка легитимной транзакции)"
)
fn_weight = st.sidebar.slider(
    "False Negative", 1, 100, 10,
    help="Стоимость пропущенного фрода"
)

available_scenarios = get_available_scenarios()
selected_scenario = st.sidebar.selectbox(
    "Стресс-сценарий", available_scenarios,
    help="Имитация аномального поведения мошенников"
)

all_models = [
    "Logistic Regression", "Random Forest v1", "Random Forest v2",
    "CatBoost v1", "CatBoost v2", "Isolation Forest"
]
selected_models = st.sidebar.multiselect(
    "Модели для сравнения", options=all_models, default=all_models,
    help="Выберите модели для оценки и сравнения"
)

uploaded_file = st.file_uploader(
    "Загрузите CSV", type=["csv"],
    help="Файл должен содержать колонки, соответствующие синтетическим данным (см. документацию)"
)

compare_button = st.sidebar.button("Запустить анализ", type="primary", help="Начать оценку моделей")

# =====================================================
# ИНИЦИАЛИЗАЦИЯ SESSION_STATE
# =====================================================

if "classic_df" not in st.session_state:
    st.session_state.classic_df = None
if "stress_df" not in st.session_state:
    st.session_state.stress_df = None
if "results_classic" not in st.session_state:
    st.session_state.results_classic = None
if "results_stress" not in st.session_state:
    st.session_state.results_stress = None
if "results_calculated" not in st.session_state:
    st.session_state.results_calculated = False
if "last_selected_models" not in st.session_state:
    st.session_state.last_selected_models = selected_models.copy()
if "last_threshold" not in st.session_state:
    st.session_state.last_threshold = threshold
if "last_fraud_ratio" not in st.session_state:
    st.session_state.last_fraud_ratio = fraud_ratio

# =====================================================
# ОСНОВНАЯ ЛОГИКА (генерация/загрузка данных при нажатии кнопки)
# =====================================================

if compare_button:
    st.toast(f"Запуск анализа со сценарием: {selected_scenario}")
    try:
        if uploaded_file is not None:
            if not uploaded_file.name.endswith(".csv"):
                st.error("Некорректное расширение файла")
                st.stop()
            classic_df = pd.read_csv(uploaded_file)
            required_columns = get_expected_columns() + ["is_fraud"]
            missing = [c for c in required_columns if c not in classic_df.columns]
            if missing:
                st.error(f"Отсутствуют колонки: {missing}")
                st.stop()
        else:
            with st.spinner("Генерация синтетических данных..."):
                classic_df = generate_fraud_subset(
                    subset_size=sample_size, full_size=2000, fraud_ratio=fraud_ratio,
                    label_noise=0.015, random_state=42, use_stratification=True
                )
        st.session_state.classic_df = classic_df
        st.session_state.stress_df = apply_stress(classic_df.copy(), selected_scenario)
        st.session_state.results_calculated = False
        # Сохраняем текущие параметры для отслеживания изменений
        st.session_state.last_selected_models = selected_models.copy()
        st.session_state.last_threshold = threshold
        st.session_state.last_fraud_ratio = fraud_ratio
    except Exception as e:
        logging.error(str(e))
        st.error(f"Ошибка при генерации/загрузке данных: {e}")

# =====================================================
# ОТОБРАЖЕНИЕ И РЕДАКТИРОВАНИЕ ДАННЫХ (если есть в сессии)
# =====================================================

if st.session_state.classic_df is not None:
    st.subheader("Редактирование датасета")
    edited_df = st.data_editor(st.session_state.classic_df, num_rows="dynamic", key="data_editor")
    # Если пользователь изменил данные, обновляем сессию и пересчитываем стресс-сценарий
    if not edited_df.equals(st.session_state.classic_df):
        st.session_state.classic_df = edited_df
        st.session_state.stress_df = apply_stress(edited_df.copy(), selected_scenario)
        st.session_state.results_calculated = False

    st.subheader("Сгенерированные данные")
    st.dataframe(st.session_state.classic_df.head(), use_container_width=True)

    # Проверка типов данных (только числовые колонки)
    feature_cols = get_expected_columns()
    wrong_types = []
    for col in feature_cols:
        if col in st.session_state.classic_df.columns and not pd.api.types.is_numeric_dtype(st.session_state.classic_df[col]):
            wrong_types.append(col)
    if wrong_types:
        st.error(f"Неверный тип данных в колонках: {wrong_types}")
        st.stop()

    # =================================================
    # ПЕРЕСЧЁТ МЕТРИК (только если изменились модели, порог или данные)
    # =================================================
    models_changed = (st.session_state.last_selected_models != selected_models)
    threshold_changed = (st.session_state.last_threshold != threshold)
    need_recalc = (not st.session_state.results_calculated) or models_changed or threshold_changed

    if need_recalc:
        X_classic = st.session_state.classic_df.drop(columns=["is_fraud"], errors="ignore")
        y_classic = st.session_state.classic_df["is_fraud"]
        X_stress = st.session_state.stress_df.drop(columns=["is_fraud"], errors="ignore")
        y_stress = st.session_state.stress_df["is_fraud"]

        all_models_dict = load_all_models()
        models = {name: all_models_dict[name] for name in selected_models if name in all_models_dict}
        st.success(f"Загружено моделей: {len(models)}")

        results_classic = []
        results_stress = []

        for name, model in models.items():
            st.info(f"Анализ модели: {name}")

            # Обработка данных в зависимости от типа модели
            if name in ["Logistic Regression", "Random Forest v1", "Random Forest v2"]:
                X_classic_processed = X_classic[feature_cols].copy()
                X_stress_processed = X_stress[feature_cols].copy()
            elif "CatBoost" in name:
                X_classic_processed = X_classic.copy()
                X_stress_processed = X_stress.copy()
                if "category" in X_classic_processed.columns:
                    X_classic_processed["category"] = X_classic_processed["category"].astype("category")
                if "category" in X_stress_processed.columns:
                    X_stress_processed["category"] = X_stress_processed["category"].astype("category")
            elif name == "Isolation Forest":
                X_classic_processed = pd.get_dummies(X_classic.copy())
                X_stress_processed = pd.get_dummies(X_stress.copy())
                X_stress_processed = X_stress_processed.reindex(columns=X_classic_processed.columns, fill_value=0)

            # Предсказания
            if hasattr(model, "predict_proba"):
                classic_proba = model.predict_proba(X_classic_processed)[:, 1]
                classic_pred = (classic_proba >= threshold).astype(int)
                stress_proba = model.predict_proba(X_stress_processed)[:, 1]
                stress_pred = (stress_proba >= threshold).astype(int)
            else:
                classic_pred = model.predict(X_classic_processed)
                stress_pred = model.predict(X_stress_processed)

            if name == "Isolation Forest":
                classic_pred = (classic_pred == -1).astype(int)
                stress_pred = (stress_pred == -1).astype(int)

            # Бизнес-стоимость
            business_cost_classic = calculate_business_cost(y_classic, classic_pred, fp_weight, fn_weight)
            business_cost_stress = calculate_business_cost(y_stress, stress_pred, fp_weight, fn_weight)

            # Метрики
            results_classic.append({
                "Модель": name,
                "Precision": round(precision_score(y_classic, classic_pred, zero_division=0), 4),
                "Recall": round(recall_score(y_classic, classic_pred, zero_division=0), 4),
                "F1": round(f1_score(y_classic, classic_pred, zero_division=0), 4),
                "Business Cost": business_cost_classic
            })
            results_stress.append({
                "Модель": name,
                "Precision": round(precision_score(y_stress, stress_pred, zero_division=0), 4),
                "Recall": round(recall_score(y_stress, stress_pred, zero_division=0), 4),
                "F1": round(f1_score(y_stress, stress_pred, zero_division=0), 4),
                "Business Cost": business_cost_stress
            })

        # Сохраняем результаты в сессию
        st.session_state.results_classic = pd.DataFrame(results_classic)
        st.session_state.results_stress = pd.DataFrame(results_stress)
        st.session_state.last_selected_models = selected_models.copy()
        st.session_state.last_threshold = threshold
        st.session_state.results_calculated = True

    # =================================================
    # ОТОБРАЖЕНИЕ РЕЗУЛЬТАТОВ (из сессии)
    # =================================================
    if st.session_state.results_classic is not None and st.session_state.results_stress is not None:
        df_classic = st.session_state.results_classic.copy()
        df_stress = st.session_state.results_stress.copy()

        # Форматирование в проценты
        for col in ["Precision", "Recall", "F1"]:
            df_classic[col] = df_classic[col].apply(lambda x: f"{x*100:.1f}%")
            df_stress[col] = df_stress[col].apply(lambda x: f"{x*100:.1f}%")

        st.subheader("Классический режим")
        st.dataframe(df_classic, use_container_width=True, hide_index=True)

        st.subheader(f"Стресс-режим: {selected_scenario}")
        st.dataframe(df_stress, use_container_width=True, hide_index=True)

        st.subheader("Сравнение моделей по F1")
        chart_df = pd.DataFrame({
            "Модель": df_classic["Модель"],
            "F1 Classic": df_classic["F1"].str.replace("%", "").astype(float),
            "F1 Stress": df_stress["F1"].str.replace("%", "").astype(float)
        })
        fig = px.bar(chart_df, x="Модель", y=["F1 Classic", "F1 Stress"], barmode="group")
        st.plotly_chart(fig, use_container_width=True)

        st.success("Анализ завершён")

# =====================================================
# ЗАГРУЗКА МОДЕЛЕЙ (кэшируется)
# =====================================================

@st.cache_resource
def load_all_models():
    models = {}
    models["Logistic Regression"] = joblib.load("models/450k_models/logistic_regression_450k.pkl")
    models["Random Forest v1"] = joblib.load("models/450k_models/random_forest_v1_450k.pkl")
    models["Random Forest v2"] = joblib.load("models/450k_models/random_forest_v2_450k.pkl")
    models["CatBoost v1"] = joblib.load("models/advanced_models/catboost_v1.pkl")
    models["CatBoost v2"] = joblib.load("models/advanced_models/catboost_v2.pkl")
    models["Isolation Forest"] = joblib.load("models/advanced_models/isolation_forest.pkl")
    return models
