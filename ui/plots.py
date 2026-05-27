"""
Графики для визуализации результатов
Будут реализованы Frontend разработчиком
"""

import streamlit as st


def plot_roc_curves(models, X, y):
    """Отображает ROC-кривые для всех моделей"""
    st.subheader("📈 ROC-кривые")
    st.info("Графики будут добавлены Frontend'ом")


def plot_confusion_matrix(y_true, y_pred, model_name):
    """Отображает тепловую карту confusion matrix"""
    st.subheader(f"🎯 Confusion Matrix - {model_name}")
    st.info("График будет добавлен Frontend'ом")


def plot_feature_importance(model, feature_names):
    """Отображает важность признаков для древовидных моделей"""
    st.subheader("📊 Важность признаков")
    st.info("График будет добавлен Frontend'ом")
