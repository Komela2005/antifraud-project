"""
# ДЕТАЛЬНЫЙ АНАЛИЗ МОДЕЛИ

st.subheader("Детальный анализ модели")

detailed_model = st.selectbox(
    "Выберите модель",
    list(models.keys())
)

if st.button("Показать анализ"):

    model = models[detailed_model]

    # Подготовка данных
    X_processed = prepare_model_data(
        detailed_model,
        model,
        X_classic
    )

    # Предсказания
    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(
            X_processed
        )[:, 1]

        predictions = (
            probabilities >= threshold
        ).astype(int)

    else:

        predictions = model.predict(X_processed)

        if detailed_model == "Isolation Forest":
            predictions = (
                predictions == -1
            ).astype(int)

        probabilities = predictions

    # =================================================
    # CONFUSION MATRIX
    # =================================================

    st.markdown("### Confusion Matrix")

    cm = confusion_matrix(
        y_classic,
        predictions
    )

    cm_df = pd.DataFrame(
        cm,
        index=["Legit", "Fraud"],
        columns=["Pred Legit", "Pred Fraud"]
    )

    fig_cm = px.imshow(
        cm_df,
        text_auto=True,
        aspect="auto",
        title=f"Confusion Matrix — {detailed_model}"
    )

    st.plotly_chart(
        fig_cm,
        use_container_width=True
    )

    # =================================================
    # РАСПРЕДЕЛЕНИЕ ВЕРОЯТНОСТЕЙ
    # =================================================

    if hasattr(model, "predict_proba"):

        st.markdown("### Распределение вероятностей")

        probability_df = pd.DataFrame({
            "Вероятность фрода": probabilities,
            "Факт": y_classic
        })

        fig_hist = px.histogram(
            probability_df,
            x="Вероятность фрода",
            color="Факт",
            nbins=40,
            barmode="overlay",
            title=f"Распределение вероятностей — {detailed_model}"
        )

        st.plotly_chart(
            fig_hist,
            use_container_width=True
        )

    # =================================================
    # ТОП-МЕТРИКИ
    # =================================================

    st.markdown("### Основные метрики")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Precision",
            f"{precision_score(y_classic, predictions, zero_division=0):.3f}"
        )

    with col2:
        st.metric(
            "Recall",
            f"{recall_score(y_classic, predictions, zero_division=0):.3f}"
        )

    with col3:
        st.metric(
            "F1",
            f"{f1_score(y_classic, predictions, zero_division=0):.3f}"
        )
"""