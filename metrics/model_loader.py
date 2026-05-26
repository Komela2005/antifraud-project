"""
Модуль загрузки моделей.
Содержит функции для загрузки моделей из папки models/ с кэшированием.
"""

import glob
import os

import joblib
import streamlit as st


@st.cache_resource
def load_all_models(models_path="models/"):
    """
    Загружает все модели из папки models/.
    Модели загружаются только 1 раз при старте приложения.
    """
    models = {}

    # Создаём папку, если её нет
    os.makedirs(models_path, exist_ok=True)

    # Ищем все модели
    model_files = glob.glob(f"{models_path}*.joblib") + glob.glob(f"{models_path}*.pkl")

    if not model_files:
        print(f"В папке {models_path} нет моделей")
        return models

    # Загружаем каждую
    for model_path in model_files:
        model_name = os.path.basename(model_path).replace(".joblib", "").replace(".pkl", "")
        try:
            models[model_name] = joblib.load(model_path)
            print(f"Загружена модель: {model_name}")
        except Exception as e:
            print(f"Ошибка загрузки {model_name}: {e}")

    print(f"📦 Всего загружено моделей: {len(models)}")
    return models


def get_available_models(models_path="models/"):
    """Возвращает список доступных моделей в папке."""
    model_files = glob.glob(f"{models_path}*.joblib") + glob.glob(f"{models_path}*.pkl")
    return [os.path.basename(f) for f in model_files]
