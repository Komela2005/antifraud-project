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
    f1_score
)

# =====================================================
# ИМПОРТ ГЕНЕРАТОРА ДАННЫХ
# =====================================================

from data_generator.generator import (
    generate_transactions,
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

# Размер выборки
sample_size = st.sidebar.slider(
    "Размер выборки",
    min_value=100,
    max_value=10000,
    value=1000,
    step=100,
    help="Количество генерируемых транзакций"
)

# Выбор модели
selected_model = st.sidebar.selectbox(
    "Выберите модель",
    [
        "logistic_regression",
        "random_forest_v1",
        "random_forest_v2"
    ],
    help=" На данный момент доступны логистическая регрессия и две версии случайного леса"
)

# Threshold
threshold = st.sidebar.slider(
    "Порог классификации",
    min_value=0.1,
    max_value=0.9,
    value=0.5,
    step=0.05,
    help="Вероятность выше этого порога считается мошенничеством"
)

# Stress scenario
available_scenarios = (
    get_available_scenarios()
)

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

# Кнопка запуска
compare_button = st.sidebar.button(
    "Запустить анализ",
    help="Нажмите для расчёта метрик на выбранных данных"
)

# =====================================================
# ИНФОРМАЦИЯ О ПАРАМЕТРАХ
# =====================================================

st.write(f"Размер выборки: {sample_size}")
st.write(f"Модель: {selected_model}")
st.write(f"Порог: {threshold}")
st.write(f"Stress scenario: {selected_scenario}")

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

            if not uploaded_file.name.endswith(
                ".csv"
            ):

                st.error(
                    "Некорректное расширение файла"
                )

                st.stop()

            classic_df = pd.read_csv(
                uploaded_file
            )

        else:
            with st.spinner("Генерируем синтетические данные, пожалуйста, подождите :)"):
            	classic_df = (
                	generate_transactions(
                    	sample_size
                	)
            	)

        # =================================================
        # СТРЕСС-СЦЕНАРИЙ
        # =================================================

        stress_df = apply_stress(
            classic_df.copy(),
            selected_scenario
        )

        # =================================================
        # ОТОБРАЖЕНИЕ DATASET
        # =================================================

        st.subheader(
            "Сгенерированные данные"
        )

        st.dataframe(
            classic_df.head()
        )

        # =================================================
        # ВАЛИДАЦИЯ КОЛОНОК
        # =================================================

        required_columns = (
            get_expected_columns()
        )

        required_columns.append(
            "is_fraud"
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

        # =================================================
        # ВАЛИДАЦИЯ ТИПОВ
        # =================================================

        wrong_types = []

        for col in get_expected_columns():

            if not pd.api.types.is_numeric_dtype(
                classic_df[col]
            ):

                wrong_types.append(col)

        if wrong_types:

            st.error(
                f"Неверный тип данных: {wrong_types}"
            )

            st.stop()

        # =================================================
        # ИНТЕРАКТИВНОЕ РЕДАКТИРОВАНИЕ
        # =================================================

        st.subheader(
            "Редактирование датасета"
        )

        edited_df = st.data_editor(
            classic_df
        )

        # =================================================
        # FEATURES / TARGET
        # =================================================

        X_classic = edited_df.drop(
            columns=["is_fraud"]
        )

        y_classic = edited_df["is_fraud"]

        X_stress = stress_df.drop(
            columns=["is_fraud"]
        )

        y_stress = stress_df["is_fraud"]

        # =================================================
        # ЗАГРУЗКА МОДЕЛИ
        # =================================================

        model_path = (
            Path("models") /
            f"{selected_model}.pkl"
        )

        model = joblib.load(
            model_path
        )

        st.success(
            f"Модель загружена: {selected_model}"
        )

        # =================================================
        # CLASSIC PREDICTIONS
        # =================================================

        if hasattr(
            model,
            "predict_proba"
        ):

            classic_proba = (
                model.predict_proba(
                    X_classic
                )[:, 1]
            )

            classic_pred = (
                classic_proba >= threshold
            ).astype(int)

        else:

            classic_pred = model.predict(
                X_classic
            )

            classic_proba = classic_pred

        # =================================================
        # STRESS PREDICTIONS
        # =================================================

        if hasattr(
            model,
            "predict_proba"
        ):

            stress_proba = (
                model.predict_proba(
                    X_stress
                )[:, 1]
            )

            stress_pred = (
                stress_proba >= threshold
            ).astype(int)

        else:

            stress_pred = model.predict(
                X_stress
            )

            stress_proba = stress_pred

        # =================================================
        # МЕТРИКИ CLASSIC
        # =================================================

        classic_precision = precision_score(
            y_classic,
            classic_pred,
            zero_division=0
        )

        classic_recall = recall_score(
            y_classic,
            classic_pred,
            zero_division=0
        )

        classic_f1 = f1_score(
            y_classic,
            classic_pred,
            zero_division=0
        )

        # =================================================
        # МЕТРИКИ STRESS
        # =================================================

        stress_precision = precision_score(
            y_stress,
            stress_pred,
            zero_division=0
        )

        stress_recall = recall_score(
            y_stress,
            stress_pred,
            zero_division=0
        )

        stress_f1 = f1_score(
            y_stress,
            stress_pred,
            zero_division=0
        )

        # =================================================
        # ТАБЛИЦА МЕТРИК
        # =================================================

        metrics_df = pd.DataFrame({

            "Метрика": [
                "Precision",
                "Recall",
                "F1"
            ],

            "Классический режим": [
                round(classic_precision, 4),
                round(classic_recall, 4),
                round(classic_f1, 4)
            ],

            "Стресс-режим": [
                round(stress_precision, 4),
                round(stress_recall, 4),
                round(stress_f1, 4)
            ]
        })

        st.subheader(
            "Таблица метрик"
        )

        st.dataframe(
            metrics_df
        )

        # =================================================
        # ГРАФИК THRESHOLD
        # =================================================

        if hasattr(
            model,
            "predict_proba"
        ):

            threshold_values = []
            precision_values = []
            recall_values = []
            f1_values = []

            for t in [
                0.1,
                0.2,
                0.3,
                0.4,
                0.5,
                0.6,
                0.7,
                0.8,
                0.9
            ]:

                temp_pred = (
                    classic_proba >= t
                ).astype(int)

                threshold_values.append(t)

                precision_values.append(
                    precision_score(
                        y_classic,
                        temp_pred,
                        zero_division=0
                    )
                )

                recall_values.append(
                    recall_score(
                        y_classic,
                        temp_pred,
                        zero_division=0
                    )
                )

                f1_values.append(
                    f1_score(
                        y_classic,
                        temp_pred,
                        zero_division=0
                    )
                )

            threshold_df = pd.DataFrame({

                "Порог": threshold_values,
                "Precision": precision_values,
                "Recall": recall_values,
                "F1": f1_values
            })

            st.subheader(
                "Метрика vs Порог"
            )

            fig_threshold = px.line(
                threshold_df,
                x="Порог",
                y=[
                    "Precision",
                    "Recall",
                    "F1"
                ]
            )

            st.plotly_chart(
                fig_threshold,
                use_container_width=True
            )

        # =================================================
        # BAR CHART
        # =================================================

        comparison_df = pd.DataFrame({

            "Метрика": [
                "Precision",
                "Recall",
                "F1"
            ],

            "Классический режим": [
                classic_precision,
                classic_recall,
                classic_f1
            ],

            "Стресс-режим": [
                stress_precision,
                stress_recall,
                stress_f1
            ]
        })

        st.subheader(
            "Сравнение режимов"
        )

        fig_bar = px.bar(
            comparison_df,
            x="Метрика",
            y=[
                "Классический режим",
                "Стресс-режим"
            ],
            barmode="group"
        )

        st.plotly_chart(
            fig_bar,
            use_container_width=True
        )

        # =================================================
        # ДЕТАЛЬНЫЙ АНАЛИЗ
        # =================================================

        if st.button(
            "Детальный анализ"
        ):

            st.subheader(
                "Подробный анализ модели"
            )

            probability_df = pd.DataFrame({

                "Вероятность мошенничества":
                    classic_proba
            })

            st.dataframe(
                probability_df.head(20)
            )

            st.subheader(
                "Распределение вероятностей"
            )

            fig_hist = px.histogram(
                probability_df,
                x="Вероятность мошенничества"
            )

            st.plotly_chart(
                fig_hist,
                use_container_width=True
            )

    # =====================================================
    # ОШИБКА ФАЙЛА МОДЕЛИ
    # =====================================================

    except FileNotFoundError:

        logging.error(
            "Файл модели не найден"
        )

        st.error(
            "Файл модели не найден"
        )

    # =====================================================
    # ОБЩИЕ ОШИБКИ
    # =====================================================

    except Exception as e:

        logging.error(str(e))

        st.error(
            f"Ошибка приложения: {e}"
        )
