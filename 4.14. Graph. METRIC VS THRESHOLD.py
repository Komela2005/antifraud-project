"""
st.subheader("Метрика vs порог")

selected_metric = st.selectbox(
    "Выберите метрику",
    ["Precision", "Recall", "F1"],
    index=2
)

# Кнопка детального анализа
show_details = st.button("Детальный анализ")

if show_details:

    threshold_values = [x / 100 for x in range(10, 95, 5)]

    metric_plot_data = []

    for name, model in models.items():

        # Подготовка данных под конкретную модель
        X_classic_processed = prepare_model_data(name, model, X_classic)

        # Получаем вероятности
        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(
                X_classic_processed
            )[:, 1]

            for current_threshold in threshold_values:

                predictions = (
                    probabilities >= current_threshold
                ).astype(int)

                precision = precision_score(
                    y_classic,
                    predictions,
                    zero_division=0
                )

                recall = recall_score(
                    y_classic,
                    predictions,
                    zero_division=0
                )

                f1 = f1_score(
                    y_classic,
                    predictions,
                    zero_division=0
                )

                metric_plot_data.append({
                    "Модель": name,
                    "Threshold": current_threshold,
                    "Precision": precision,
                    "Recall": recall,
                    "F1": f1
                })

    threshold_df = pd.DataFrame(metric_plot_data)

    # Построение графика
    fig_threshold = px.line(
        threshold_df,
        x="Threshold",
        y=selected_metric,
        color="Модель",
        markers=True,
        title=f"{selected_metric} vs Threshold"
    )

    st.plotly_chart(
        fig_threshold,
        use_container_width=True
    )
"""