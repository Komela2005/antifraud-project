# =====================================================
# ИМПОРТ БИБЛИОТЕК
# =====================================================

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

def calculate_business_cost(
    y_true,
    y_pred,
    fp_weight,
    fn_weight
):
    """
    Расчёт стоимости ошибок модели.
    
    FP = ложная тревога
    FN = пропущенный фрод
    """

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred
    ).ravel()

    total_cost = (
        fp * fp_weight
        +
        fn * fn_weight
    )

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

# =====================================================
# РАЗМЕР ВЫБОРКИ
# =====================================================

sample_size = st.sidebar.slider(
    "Размер выборки",
    min_value=100,
    max_value=2000,
    value=1000,
    step=100
)

# =====================================================
# ПОРОГ КЛАССИФИКАЦИИ
# =====================================================

threshold = st.sidebar.slider(
    "Порог классификации",
    min_value=0.1,
    max_value=0.9,
    value=0.5,
    step=0.05
)

# =====================================================
# ДОЛЯ ФРОДА
# =====================================================

fraud_ratio = st.sidebar.slider(
    "Доля мошеннических транзакций",
    min_value=0.01,
    max_value=0.30,
    value=0.05,
    step=0.01,
    format="%.2f"
)

# =====================================================
# COST MATRIX
# =====================================================

st.sidebar.markdown("---")
st.sidebar.subheader("Стоимость ошибок")

fp_weight = st.sidebar.slider(
    "False Positive",
    min_value=1,
    max_value=100,
    value=1
)

fn_weight = st.sidebar.slider(
    "False Negative",
    min_value=1,
    max_value=100,
    value=10
)

# =====================================================
# STRESS SCENARIO
# =====================================================

available_scenarios = get_available_scenarios()

selected_scenario = st.sidebar.selectbox(
    "Стресс-сценарий",
    available_scenarios
)

# =====================================================
# СПИСОК МОДЕЛЕЙ
# =====================================================

all_models = [
    "Logistic Regression",
    "Random Forest v1",
    "Random Forest v2",
    "CatBoost v1",
    "CatBoost v2",
    "Isolation Forest"
]

selected_models = st.sidebar.multiselect(
    "Модели для сравнения",
    options=all_models,
    default=all_models
)

# =====================================================
# ЗАГРУЗКА CSV
# =====================================================

uploaded_file = st.file_uploader(
    "Загрузите CSV",
    type=["csv"]
)

# =====================================================
# КНОПКА ЗАПУСКА
# =====================================================

compare_button = st.sidebar.button(
    "Запустить анализ",
    type="primary"
)

# =====================================================
# ЗАГРУЗКА МОДЕЛЕЙ
# =====================================================

@st.cache_resource
def load_all_models():

    models = {}

    models["Logistic Regression"] = joblib.load(
        "models/450k_models/logistic_regression_450k.pkl"
    )

    models["Random Forest v1"] = joblib.load(
        "models/450k_models/random_forest_v1_450k.pkl"
    )

    models["Random Forest v2"] = joblib.load(
        "models/450k_models/random_forest_v2_450k.pkl"
    )

    models["CatBoost v1"] = joblib.load(
        "models/advanced_models/catboost_v1.pkl"
    )

    models["CatBoost v2"] = joblib.load(
        "models/advanced_models/catboost_v2.pkl"
    )

    models["Isolation Forest"] = joblib.load(
        "models/advanced_models/isolation_forest.pkl"
    )

    return models

# =====================================================
# ОСНОВНАЯ ЛОГИКА
# =====================================================

if compare_button:

    try:

        # =================================================
        # ГЕНЕРАЦИЯ ИЛИ ЗАГРУЗКА ДАННЫХ
        # =================================================

        if uploaded_file is not None:

            if not uploaded_file.name.endswith(".csv"):

                st.error(
                    "Некорректное расширение файла"
                )

                st.stop()

            classic_df = pd.read_csv(
                uploaded_file
            )

            required_columns = (
                get_expected_columns()
                +
                ["is_fraud"]
            )

            missing_columns = [
                col
                for col in required_columns
                if col not in classic_df.columns
            ]

            if missing_columns:

                st.error(
                    f"Отсутствуют колонки: {missing_columns}"
                )

                st.stop()

        else:

            with st.spinner(
                "Генерация данных..."
            ):

                classic_df = generate_fraud_subset(
                    subset_size=sample_size,
                    full_size=2000,
                    fraud_ratio=fraud_ratio,
                    label_noise=0.015,
                    random_state=42,
                    use_stratification=True
                )

        # =================================================
        # STRESS SCENARIO
        # =================================================

        stress_df = apply_stress(
            classic_df.copy(),
            selected_scenario
        )

        # =================================================
        # DATAFRAME
        # =================================================

        st.subheader(
            "Сгенерированные данные"
        )

        st.dataframe(
            classic_df.head(),
            use_container_width=True
        )

        # =================================================
        # ПРОВЕРКА ТИПОВ
        # =================================================

        feature_cols = get_expected_columns()

        wrong_types = []

        for col in feature_cols:

            if (
                col in classic_df.columns
                and
                not pd.api.types.is_numeric_dtype(
                    classic_df[col]
                )
            ):

                wrong_types.append(col)

        if wrong_types:

            st.error(
                f"Неверный тип данных: {wrong_types}"
            )

            st.stop()

        # =================================================
        # РЕДАКТИРОВАНИЕ DATASET
        # =================================================

        st.subheader(
            "Редактирование датасета"
        )

        edited_df = st.data_editor(
            classic_df,
            num_rows="dynamic"
        )

        # =================================================
        # FEATURES / TARGET
        # =================================================

        X_classic = edited_df.drop(
            columns=["is_fraud"],
            errors="ignore"
        )

        y_classic = edited_df[
            "is_fraud"
        ]

        X_stress = stress_df.drop(
            columns=["is_fraud"],
            errors="ignore"
        )

        y_stress = stress_df[
            "is_fraud"
        ]

        # =================================================
        # ЗАГРУЗКА МОДЕЛЕЙ
        # =================================================

        all_models_dict = load_all_models()

        models = {
            name: all_models_dict[name]
            for name in selected_models
            if name in all_models_dict
        }

        st.success(
            f"Загружено моделей: {len(models)}"
        )

        # =================================================
        # РЕЗУЛЬТАТЫ
        # =================================================

        results_classic = []
        results_stress = []

        # =================================================
        # ЦИКЛ ПО МОДЕЛЯМ
        # =================================================

        for name, model in models.items():

            st.info(
                f"Анализ модели: {name}"
            )

            # =============================================
            # LOGISTIC + RANDOM FOREST
            # =============================================

            if name in [
                "Logistic Regression",
                "Random Forest v1",
                "Random Forest v2"
            ]:

                X_classic_processed = X_classic[
                    feature_cols
                ].copy()

                X_stress_processed = X_stress[
                    feature_cols
                ].copy()

            # =============================================
            # CATBOOST
            # =============================================

            elif "CatBoost" in name:

                X_classic_processed = (
                    X_classic.copy()
                )

                X_stress_processed = (
                    X_stress.copy()
                )

                if "category" in X_classic_processed.columns:

                    X_classic_processed[
                        "category"
                    ] = (
                        X_classic_processed[
                            "category"
                        ].astype("category")
                    )

                if "category" in X_stress_processed.columns:

                    X_stress_processed[
                        "category"
                    ] = (
                        X_stress_processed[
                            "category"
                        ].astype("category")
                    )

            # =============================================
            # ISOLATION FOREST
            # =============================================

            elif name == "Isolation Forest":

                X_classic_processed = (
                    pd.get_dummies(
                        X_classic.copy()
                    )
                )

                X_stress_processed = (
                    pd.get_dummies(
                        X_stress.copy()
                    )
                )

                # =========================================
                # ВЫРАВНИВАНИЕ КОЛОНОК
                # =========================================

                X_stress_processed = (
                    X_stress_processed.reindex(
                        columns=X_classic_processed.columns,
                        fill_value=0
                    )
                )

            # =============================================
            # PREDICT CLASSIC
            # =============================================

            if hasattr(
                model,
                "predict_proba"
            ):

                classic_proba = (
                    model.predict_proba(
                        X_classic_processed
                    )[:, 1]
                )

                classic_pred = (
                    classic_proba >= threshold
                ).astype(int)

            else:

                classic_pred = model.predict(
                    X_classic_processed
                )

            # =============================================
            # ISOLATION FOREST FIX
            # =============================================

            if name == "Isolation Forest":

                classic_pred = (
                    classic_pred == -1
                ).astype(int)

            # =============================================
            # PREDICT STRESS
            # =============================================

            if hasattr(
                model,
                "predict_proba"
            ):

                stress_proba = (
                    model.predict_proba(
                        X_stress_processed
                    )[:, 1]
                )

                stress_pred = (
                    stress_proba >= threshold
                ).astype(int)

            else:

                stress_pred = model.predict(
                    X_stress_processed
                )

            # =============================================
            # ISOLATION FOREST FIX
            # =============================================

            if name == "Isolation Forest":

                stress_pred = (
                    stress_pred == -1
                ).astype(int)

            # =============================================
            # BUSINESS COST
            # =============================================

            business_cost_classic = (
                calculate_business_cost(
                    y_classic,
                    classic_pred,
                    fp_weight,
                    fn_weight
                )
            )

            business_cost_stress = (
                calculate_business_cost(
                    y_stress,
                    stress_pred,
                    fp_weight,
                    fn_weight
                )
            )

            # =============================================
            # METRICS CLASSIC
            # =============================================

            results_classic.append({

                "Модель": name,

                "Precision": round(
                    precision_score(
                        y_classic,
                        classic_pred,
                        zero_division=0
                    ),
                    4
                ),

                "Recall": round(
                    recall_score(
                        y_classic,
                        classic_pred,
                        zero_division=0
                    ),
                    4
                ),

                "F1": round(
                    f1_score(
                        y_classic,
                        classic_pred,
                        zero_division=0
                    ),
                    4
                ),

                "Business Cost": (
                    business_cost_classic
                )
            })

            # =============================================
            # METRICS STRESS
            # =============================================

            results_stress.append({

                "Модель": name,

                "Precision": round(
                    precision_score(
                        y_stress,
                        stress_pred,
                        zero_division=0
                    ),
                    4
                ),

                "Recall": round(
                    recall_score(
                        y_stress,
                        stress_pred,
                        zero_division=0
                    ),
                    4
                ),

                "F1": round(
                    f1_score(
                        y_stress,
                        stress_pred,
                        zero_division=0
                    ),
                    4
                ),

                "Business Cost": (
                    business_cost_stress
                )
            })

        # =================================================
        # DATAFRAME METRICS
        # =================================================

        df_classic = pd.DataFrame(
            results_classic
        )

        df_stress = pd.DataFrame(
            results_stress
        )

        # =================================================
        # ПРОЦЕНТЫ
        # =================================================

        for col in [
            "Precision",
            "Recall",
            "F1"
        ]:

            df_classic[col] = (
                df_classic[col]
                .apply(
                    lambda x:
                    f"{x*100:.1f}%"
                )
            )

            df_stress[col] = (
                df_stress[col]
                .apply(
                    lambda x:
                    f"{x*100:.1f}%"
                )
            )

        # =================================================
        # ТАБЛИЦЫ
        # =================================================

        st.subheader(
            "Классический режим"
        )

        st.dataframe(
            df_classic,
            use_container_width=True,
            hide_index=True
        )

        st.subheader(
            f"Стресс-режим: {selected_scenario}"
        )

        st.dataframe(
            df_stress,
            use_container_width=True,
            hide_index=True
        )

        # =================================================
        # BAR CHART
        # =================================================

        st.subheader(
            "Сравнение моделей по F1"
        )

        chart_df = pd.DataFrame({

            "Модель":
            df_classic["Модель"],

            "F1 Classic":
            df_classic["F1"]
            .str.replace("%", "")
            .astype(float),

            "F1 Stress":
            df_stress["F1"]
            .str.replace("%", "")
            .astype(float)
        })

        fig = px.bar(
            chart_df,
            x="Модель",
            y=[
                "F1 Classic",
                "F1 Stress"
            ],
            barmode="group"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # =================================================
        # УСПЕХ
        # =================================================

        st.success(
            "Анализ завершён"
        )

    # =====================================================
    # FILE NOT FOUND
    # =====================================================

    except FileNotFoundError as e:

        logging.error(str(e))

        st.error(
            f"Файл модели не найден: {e}"
        )

    # =====================================================
    # ОБЩАЯ ОШИБКА
    # =====================================================

    except Exception as e:

        logging.error(str(e))

        st.error(
            f"Ошибка приложения: {e}"
        )        st.error(f"Ошибка приложения: {e}")