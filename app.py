import logging
 
import joblib
import pandas as pd
import plotly.express as px
import streamlit as st
from database import init_db, create_experiment, save_model_results, finish_experiment
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
# ФУНКЦИИ ЗАГРУЗКИ ДАННЫХ
# =====================================================
 
def load_or_generate_data(uploaded_file, sample_size, fraud_ratio):
    """Загружает пользовательский CSV или генерирует синтетику"""
    if uploaded_file is not None:
        if not uploaded_file.name.endswith(".csv"):
            st.error("Некорректное расширение файла")
            return None, None
 
        df = pd.read_csv(uploaded_file)
 
        from metrics.validator import validate_csv
        is_valid, errors, warnings = validate_csv(df, require_target=False)
 
        for warning in warnings:
            st.warning(warning)
 
        if not is_valid:
            for error in errors:
                st.error(error)
            return None, None
 
        return df, "uploaded"
    else:
        with st.spinner("Генерация синтетических данных..."):
            df = generate_fraud_subset(
                subset_size=sample_size, full_size=2000, fraud_ratio=fraud_ratio,
                label_noise=0.015, random_state=42, use_stratification=True
            )
        return df, "synthetic"
 
# =====================================================
# ФУНКЦИИ ПОДГОТОВКИ ДАННЫХ ДЛЯ МОДЕЛЕЙ
# =====================================================
 
def prepare_data_for_lr_rf(X, feature_cols):
    # Добавляем недостающие колонки со значением 0
    return X.reindex(columns=feature_cols, fill_value=0)
 
def prepare_data_for_catboost(X):
    """Подготовка данных для CatBoost"""
    X = X.copy()
    if "category" in X.columns:
        X["category"] = X["category"].astype("category")
    return X
 
 
def prepare_data_for_iforest(X):
    """Подготовка данных для Isolation Forest (one-hot encoding)"""
    return pd.get_dummies(X.copy())
 
 
def prepare_model_data(model_name, model, X):
    """
    Универсальная функция подготовки данных для любой модели.
    X — датафрейм без колонки is_fraud.
    """
    if model_name in ["Logistic Regression", "Random Forest v1", "Random Forest v2"]:
        X = X.copy()
    
        # Если есть минутные признаки, создаём часовые (для совместимости со старыми моделями)
        if 'transaction_minute' in X.columns and 'transaction_hour' not in X.columns:
            X['transaction_hour'] = X['transaction_minute'] // 60
        if 'typical_minute_client' in X.columns and 'typical_hour_client' not in X.columns:
            X['typical_hour_client'] = X['typical_minute_client'] // 60
        
        # Для старых моделей могли отсутствовать time_deviation_min и is_contactless – добавим их с нулём
        if 'time_deviation_min' not in X.columns:
            X['time_deviation_min'] = 0
        if 'is_contactless' not in X.columns:
            X['is_contactless'] = 0
        
        # Получаем ожидаемые колонки из модели
        if hasattr(model, "feature_names_in_"):
            expected_cols = list(model.feature_names_in_)
        else:
            # fallback – используем старый список (например, get_expected_columns из генератора, который ещё не обновлён)
            from data_generator.generator import get_expected_columns as old_expected
            expected_cols = [col for col in old_expected() if col != 'category']
            
        return prepare_data_for_lr_rf(X, expected_cols)
 
    elif "CatBoost" in model_name:
        X = X.copy()
        # Конвертируем минуты в часы, если модель ожидает часы
        if 'transaction_minute' in X.columns and 'transaction_hour' not in X.columns:
            X['transaction_hour'] = X['transaction_minute'] // 60
        if 'typical_minute_client' in X.columns and 'typical_hour_client' not in X.columns:
            X['typical_hour_client'] = X['typical_minute_client'] // 60
        # Приводим категориальный признак
        if "category" in X.columns:
            X["category"] = X["category"].astype("category")
        # Приводим к ожидаемым колонкам модели (если есть)
        if hasattr(model, "feature_names_in_"):
            expected_cols = list(model.feature_names_in_)
        return prepare_data_for_catboost(X)
 
    elif model_name == "Isolation Forest":
        X_processed = prepare_data_for_iforest(X)
        if hasattr(model, "feature_names_in_"):
            expected_cols = list(model.feature_names_in_)
            X_processed = X_processed.reindex(columns=expected_cols, fill_value=0)
        return X_processed
 
    return X.copy()
 
 
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
init_db()
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
# SIDEBAR
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
 
defaults = {
    "classic_df_full": None,   # полный df (признаки + is_fraud) — для отображения и редактирования
    "X_classic": None,         # только признаки (без is_fraud)
    "y_classic": None,         # целевая переменная классика
    "stress_df_full": None,    # полный стресс-df (признаки + is_fraud)
    "X_stress": None,          # только признаки стресса
    "y_stress": None,          # целевая переменная стресса
    "results_classic": None,
    "results_stress": None,
    "results_calculated": False,
    "last_selected_models": selected_models.copy(),
    "last_threshold": threshold,
    "last_fraud_ratio": fraud_ratio,
    "data_valid": True,
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val
 
# =====================================================
# ОСНОВНАЯ ЛОГИКА — загрузка/генерация данных
# =====================================================
 
if compare_button:
    st.toast(f"Запуск анализа со сценарием: {selected_scenario}")
    try:
        from metrics.validator import prepare_data_for_prediction, get_column_info, validate_csv
 
        # 1. Загружаем / генерируем полный df (с is_fraud)
        classic_df_full, source = load_or_generate_data(uploaded_file, sample_size, fraud_ratio)
        if classic_df_full is None:
            st.stop()
 
        # 2. Сохраняем целевую переменную ДО удаления
        y_classic = (
            classic_df_full["is_fraud"].copy()
            if "is_fraud" in classic_df_full.columns
            else None
        )
 
        # 3. Готовим признаки (prepare_data_for_prediction удаляет is_fraud)
        X_classic = prepare_data_for_prediction(classic_df_full)
 
        # 4. Собираем обратно полный df для отображения и редактирования
        classic_df_display = X_classic.copy()
        if y_classic is not None:
            classic_df_display["is_fraud"] = y_classic.values
 
        # 5. Сохраняем в session_state
        st.session_state.classic_df_full = classic_df_display
        st.session_state.X_classic = X_classic
        st.session_state.y_classic = y_classic
 
        # 6. Применяем стресс к полному df (чтобы is_fraud пережил трансформацию)
        stress_df_full = apply_stress(classic_df_display.copy(), selected_scenario)
        y_stress = (
            stress_df_full["is_fraud"].copy()
            if "is_fraud" in stress_df_full.columns
            else None
        )
        X_stress = prepare_data_for_prediction(stress_df_full)
 
        st.session_state.stress_df_full = stress_df_full
        st.session_state.X_stress = X_stress
        st.session_state.y_stress = y_stress
 
        st.session_state.results_calculated = False
        st.session_state.last_selected_models = selected_models.copy()
        st.session_state.last_threshold = threshold
        st.session_state.last_fraud_ratio = fraud_ratio
 
        # Сводка
        col_info = get_column_info()
        st.success(f"Данные загружены! {len(X_classic)} строк, {len(X_classic.columns)} признаков.")
 
        with st.expander("Информация о признаках"):
            st.markdown(f"**Всего признаков:** {col_info['total_count']}")
            st.markdown("**Основные признаки:**")
            for name, desc in list(col_info['sample_types'].items())[:5]:
                st.markdown(f"- `{name}`: {desc}")
 
    except Exception as e:
        logging.error(str(e))
        st.error(f"Ошибка при генерации/загрузке данных: {e}")
 
# =====================================================
# ОТОБРАЖЕНИЕ И РЕДАКТИРОВАНИЕ ДАННЫХ
# =====================================================
 
if st.session_state.classic_df_full is not None:
    st.subheader("Редактирование датасета")
    edited_df = st.data_editor(
        st.session_state.classic_df_full,
        num_rows="dynamic",
        key="data_editor"
    )
 
    if not edited_df.equals(st.session_state.classic_df_full):
        from metrics.validator import validate_csv, prepare_data_for_prediction
 
        require_target = "is_fraud" in edited_df.columns
        is_valid, errors, warnings = validate_csv(edited_df, require_target=require_target)
 
        for w in warnings:
            st.warning(w)
 
        if is_valid:
            st.success("Отредактированные данные прошли валидацию")
 
            # Сохраняем полный отредактированный df
            st.session_state.classic_df_full = edited_df.copy()
 
            # Разделяем заново
            y_new = (
                edited_df["is_fraud"].copy()
                if "is_fraud" in edited_df.columns
                else None
            )
            X_new = prepare_data_for_prediction(edited_df)
            st.session_state.X_classic = X_new
            st.session_state.y_classic = y_new
 
            # Обновляем стресс
            stress_full = apply_stress(edited_df.copy(), selected_scenario)
            y_stress_new = (
                stress_full["is_fraud"].copy()
                if "is_fraud" in stress_full.columns
                else None
            )
            X_stress_new = prepare_data_for_prediction(stress_full)
            st.session_state.stress_df_full = stress_full
            st.session_state.X_stress = X_stress_new
            st.session_state.y_stress = y_stress_new
 
            st.session_state.results_calculated = False
            st.session_state.data_valid = True
        else:
            for error in errors:
                st.error(error)
            st.warning("Отредактированные данные содержат ошибки. Модели не будут запущены до исправления.")
            st.session_state.data_valid = False
 
    st.subheader("Сгенерированные данные")
    st.dataframe(st.session_state.classic_df_full.head(), use_container_width=True)
 
    # =================================================
    # ПЕРЕСЧЁТ МЕТРИК
    # =================================================
    models_changed = (st.session_state.last_selected_models != selected_models)
    threshold_changed = (st.session_state.last_threshold != threshold)
    need_recalc = (
        not st.session_state.results_calculated
        or models_changed
        or threshold_changed
    )
 
    if need_recalc and st.session_state.data_valid:
        # Берём уже разделённые данные из session_state
        X_classic = st.session_state.X_classic
        y_classic = st.session_state.y_classic
        X_stress = st.session_state.X_stress
        y_stress = st.session_state.y_stress
 
        if y_classic is None:
            st.error(
                "В данных отсутствует колонка 'is_fraud' (целевая переменная). "
                "Для синтетических данных это ошибка генерации. "
                "Для пользовательских CSV — добавьте колонку с метками 0/1."
            )
            st.stop()
 
        if y_stress is None:
            st.error("В стресс-данных отсутствует колонка 'is_fraud'.")
            st.stop()
 
        all_models_dict = load_all_models()
        models = {
            name: all_models_dict[name]
            for name in selected_models
            if name in all_models_dict
        }
        st.success(f"Загружено моделей: {len(models)}")
 
        results_classic = []
        results_stress = []
 
        for name, model in models.items():
            st.info(f"Анализ модели: {name}")
 
            try:
                X_classic_processed = prepare_model_data(name, model, X_classic)
                X_stress_processed = prepare_model_data(name, model, X_stress)
            except ValueError as e:
                st.error(str(e))
                st.stop()
 
            if hasattr(model, "predict_proba"):
                classic_proba = model.predict_proba(X_classic_processed)[:, 1]
                classic_pred = (classic_proba >= threshold).astype(int)
                stress_proba = model.predict_proba(X_stress_processed)[:, 1]
                stress_pred = (stress_proba >= threshold).astype(int)
            else:
                classic_pred = model.predict(X_classic_processed)
                stress_pred = model.predict(X_stress_processed)
 
            # Isolation Forest возвращает -1 для аномалий
            if name == "Isolation Forest":
                classic_pred = (classic_pred == -1).astype(int)
                stress_pred = (stress_pred == -1).astype(int)
 
            results_classic.append({
                "Модель": name,
                "Precision": round(precision_score(y_classic, classic_pred, zero_division=0), 4),
                "Recall": round(recall_score(y_classic, classic_pred, zero_division=0), 4),
                "F1": round(f1_score(y_classic, classic_pred, zero_division=0), 4),
                "Business Cost": calculate_business_cost(y_classic, classic_pred, fp_weight, fn_weight),
            })
            results_stress.append({
                "Модель": name,
                "Precision": round(precision_score(y_stress, stress_pred, zero_division=0), 4),
                "Recall": round(recall_score(y_stress, stress_pred, zero_division=0), 4),
                "F1": round(f1_score(y_stress, stress_pred, zero_division=0), 4),
                "Business Cost": calculate_business_cost(y_stress, stress_pred, fp_weight, fn_weight),
            })
 
        st.session_state.results_classic = pd.DataFrame(results_classic)
        st.session_state.results_stress = pd.DataFrame(results_stress)
        st.session_state.last_selected_models = selected_models.copy()
        st.session_state.last_threshold = threshold
        st.session_state.results_calculated = True
 
    # =================================================
    # ОТОБРАЖЕНИЕ РЕЗУЛЬТАТОВ И ЛОГИРОВАНИЕ В БД
    # =================================================
    if (
        st.session_state.results_classic is not None
        and st.session_state.results_stress is not None
    ):
        df_classic = st.session_state.results_classic.copy()
        df_stress = st.session_state.results_stress.copy()
 
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
            "F1 Stress": df_stress["F1"].str.replace("%", "").astype(float),
        })
        fig = px.bar(chart_df, x="Модель", y=["F1 Classic", "F1 Stress"], barmode="group")
        st.plotly_chart(fig, use_container_width=True)
 
        # ЛОГИРОВАНИЕ В БД
        try:
            exp_id = create_experiment(
                sample_size=sample_size,
                threshold=threshold,
                fraud_ratio=fraud_ratio,
                stress_scenario=selected_scenario,
                models_used=selected_models,
            )
            for _, row in df_classic.iterrows():
                # Проверка business_cost на NaN
                business_cost = row["Business Cost"]
                if pd.isna(business_cost) or business_cost is None:
                    business_cost = 0
                
                save_model_results(
                    exp_id=exp_id,
                    model_name=row["Модель"],
                    mode="classic",
                    precision=float(row["Precision"].rstrip("%")) / 100,
                    recall=float(row["Recall"].rstrip("%")) / 100,
                    f1=float(row["F1"].rstrip("%")) / 100,
                    business_cost=business_cost,
                )
            for _, row in df_stress.iterrows():
                # Проверка business_cost на NaN
                business_cost = row["Business Cost"]
                if pd.isna(business_cost) or business_cost is None:
                    business_cost = 0
                
                save_model_results(
                    exp_id=exp_id,
                    model_name=row["Модель"],
                    mode="stress",
                    precision=float(row["Precision"].rstrip("%")) / 100,
                    recall=float(row["Recall"].rstrip("%")) / 100,
                    f1=float(row["F1"].rstrip("%")) / 100,
                    business_cost=business_cost,
                )
            finish_experiment(exp_id)
            st.success(f"Эксперимент сохранён в БД (ID: {exp_id})")
        except Exception as db_err:
            logging.error(f"DB error: {db_err}")
            st.warning("Не удалось сохранить результаты в базу данных")
 
        st.success("Анализ завершён")
 