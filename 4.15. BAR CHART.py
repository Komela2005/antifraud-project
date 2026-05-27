#  СРАВНЕНИЯ МОДЕЛЕЙ

st.subheader("Сравнение моделей")

metric_for_bar = st.selectbox(
    "Метрика для сравнения",
    ["Precision", "Recall", "F1"],
    key="bar_metric"
)

comparison_mode = st.radio(
    "Режим сравнения",
    ["Classic", "Stress"],
    horizontal=True
)

# Выбор нужной таблицы
if comparison_mode == "Classic":
    chart_source = st.session_state.results_classic.copy()
else:
    chart_source = st.session_state.results_stress.copy()

# Строим график
fig_bar = px.bar(
    chart_source,
    x="Модель",
    y=metric_for_bar,
    color="Модель",
    text=metric_for_bar,
    title=f"{metric_for_bar} — сравнение моделей ({comparison_mode})"
)

fig_bar.update_traces(
    texttemplate='%{text:.3f}',
    textposition='outside'
)

st.plotly_chart(
    fig_bar,
    use_container_width=True
)