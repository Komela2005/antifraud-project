# =====================================================
# ИМПОРТ БИБЛИОТЕК
# =====================================================

# logging:
# используется для записи ошибок в app.log
import logging

# pathlib.Path:
# удобная работа с путями к файлам
from pathlib import Path

# joblib:
# загрузка .pkl моделей sklearn
import joblib

# pandas:
# работа с таблицами и DataFrame
import pandas as pd

# plotly:
# интерактивные графики
import plotly.express as px

# streamlit:
# frontend framework для ML dashboard
import streamlit as st

# Метрики качества классификации
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score
)

# =====================================================
# ИМПОРТ ГЕНЕРАТОРА ДАННЫХ
# =====================================================

# generate_transactions:
# генерирует synthetic fraud dataset
#
# get_expected_columns:
# возвращает список ожидаемых признаков
from data_generator.generator import (
    generate_transactions,
    get_expected_columns
)

# =====================================================
# ИМПОРТ STRESS SCENARIOS
# =====================================================

# apply_stress:
# применяет stress scenario к датасету
#
# get_available_scenarios:
# возвращает список доступных stress-сценариев
from data_generator.stress_scenarios import (
    apply_stress,
    get_available_scenarios
)

# =====================================================
# НАСТРОЙКА ЛОГИРОВАНИЯ
# =====================================================

# Все ошибки будут записываться:
# app.log
logging.basicConfig(
    filename="app.log",
    level=logging.ERROR,
    format="%(asctime)s %(levelname)s %(message)s"
)

# =====================================================
# НАСТРОЙКА СТРАНИЦЫ STREAMLIT
# =====================================================

# page_title:
# название вкладки браузера
#
# layout="wide":
# широкий режим страницы
st.set_page_config(
    page_title="Система антифрода",
    layout="wide"
)

# Главный заголовок страницы
st.title("Система антифрода")

# =====================================================
# SIDEBAR
# =====================================================

# Боковая панель с настройками
st.sidebar.header("Настройки")

# =====================================================
# SLIDER: размер выборки
# =====================================================

# Пользователь выбирает:
# сколько строк будет сгенерировано
sample_size = st.sidebar.slider(
    "Размер выборки",
    min_value=100,
    max_value=10000,
    value=1000,
    step=100
)

# =====================================================
# SELECTBOX: выбор модели
# =====================================================

# Пользователь выбирает:
# какую ML-модель использовать
selected_model = st.sidebar.selectbox(
    "Выберите модель",
    [
        "logistic_regression",
        "random_forest_v1",
        "random_forest_v2"
    ]
)

# =====================================================
# SLIDER: threshold
# =====================================================

# Порог классификации:
# влияет на precision/recall/F1
threshold = st.sidebar.slider(
    "Порог классификации",
    min_value=0.1,
    max_value=0.9,
    value=0.5,
    step=0.05
)

# =====================================================
# SELECTBOX: stress scenario
# =====================================================

# Получаем список stress-сценариев
available_scenarios = get_available_scenarios()

# Пользователь выбирает:
# какой стресс-сценарий применить
selected_scenario = st.sidebar.selectbox(
    "Стресс-сценарий",
    available_scenarios
)

# =====================================================
# BUTTON
# =====================================================

# Кнопка запуска анализа
compare_button = st.sidebar.button(
    "Запустить анализ"
)

# =====================================================
# ОТОБРАЖЕНИЕ ВЫБРАННЫХ ПАРАМЕТРОВ
# =====================================================

st.write(f"Размер выборки: {sample_size}")
st.write(f"Выбранная модель: {selected_model}")
st.write(f"Порог: {threshold}")
st.write(f"Стресс-сценарий: {selected_scenario}")

# =====================================================
# ОСНОВНАЯ ЛОГИКА
# =====================================================

# Код выполняется:
# только после нажатия кнопки
if compare_button:

    try:

        # =================================================
        # ГЕНЕРАЦИЯ CLASSIC DATASET
        # =================================================

        # Создаем synthetic dataset
        classic_df = generate_transactions(
            sample_size
        )

        # =================================================
        # ГЕНЕРАЦИЯ STRESS DATASET
        # =================================================

        # Создаем стресс-версию датасета
        stress_df = apply_stress(
            classic_df.copy(),
            selected_scenario
        )

        # =================================================
        # ПОКАЗЫВАЕМ DATASET
        # =================================================

        st.subheader("Сгенерированные данные")

        # head():
        # первые 5 строк таблицы
        st.dataframe(
            classic_df.head()
        )

        # =================================================
        # ВАЛИДАЦИЯ DATASET
        # =================================================

        # Получаем ожидаемые колонки
        required_columns = (
            get_expected_columns()
        )

        # Добавляем target column
        required_columns.append(
            "is_fraud"
        )

        # Ищем отсутствующие колонки
        missing_columns = [
            col
            for col in required_columns
            if col not in classic_df.columns
        ]

        # Если колонки отсутствуют
        if missing_columns:

            st.error(
                f"Отсутствуют колонки: {missing_columns}"
            )

            # Останавливаем выполнение
            st.stop()

        # =================================================
        # ИНТЕРАКТИВНОЕ РЕДАКТИРОВАНИЕ
        # =================================================

        st.subheader(
            "Редактирование датасета"
        )

        # Пользователь может:
        # менять ячейки прямо в UI
        edited_df = st.data_editor(
            classic_df
        )

        # =================================================
        # FEATURES / TARGET
        # =================================================

        # X:
        # признаки модели
        X_classic = edited_df.drop(
            columns=["is_fraud"]
        )

        # y:
        # target column
        y_classic = edited_df["is_fraud"]

        # Stress features
        X_stress = stress_df.drop(
            columns=["is_fraud"]
        )

        # Stress target
        y_stress = stress_df["is_fraud"]

        # =================================================
        # ЗАГРУЗКА МОДЕЛИ
        # =================================================

        # Формируем путь:
        # models/random_forest_v1.pkl
        model_path = (
            Path("models") /
            f"{selected_model}.pkl"
        )

        # Загружаем модель
        model = joblib.load(model_path)

        st.success(
            f"Модель загружена: {selected_model}"
        )

        # =================================================
        # CLASSIC PREDICTIONS
        # =================================================

        # Проверяем:
        # умеет ли модель predict_proba
        if hasattr(model, "predict_proba"):

            # Получаем вероятности fraud
            classic_proba = (
                model.predict_proba(
                    X_classic
                )[:, 1]
            )

            # Применяем threshold
            classic_pred = (
                classic_proba >= threshold
            ).astype(int)

        else:

            # Если predict_proba нет
            classic_pred = model.predict(
                X_classic
            )

            classic_proba = classic_pred

        # =================================================
        # STRESS PREDICTIONS
        # =================================================

        if hasattr(model, "predict_proba"):

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
        # ВЫЧИСЛЕНИЕ METRICS
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

        st.dataframe(metrics_df)

        # =================================================
        # BAR CHART
        # =================================================

        st.subheader(
            "Сравнение режимов"
        )

        fig_bar = px.bar(
            metrics_df,
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

    # =====================================================
    # FILE NOT FOUND
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

        # Записываем ошибку в app.log
        logging.error(str(e))

        # Показываем ошибку пользователю
        st.error(
            f"Ошибка приложения: {e}"
        )
