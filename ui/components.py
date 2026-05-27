"""
UI компоненты для Streamlit
Будут реализованы Frontend разработчиком
"""

import streamlit as st


def show_metrics_table(metrics_df):
    """Отображает таблицу с метриками моделей"""
    st.subheader("📊 Сравнение метрик")
    st.dataframe(metrics_df)


def show_cost_matrix(cost_fp, cost_fn):
    """Отображает и позволяет редактировать cost matrix"""
    st.subheader("💰 Настройка стоимости ошибок")
    st.write(f"False Positive стоимость: {cost_fp}")
    st.write(f"False Negative стоимость: {cost_fn}")
