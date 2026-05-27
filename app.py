import logging

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics import (confusion_matrix, f1_score, precision_score,
                             recall_score)

from data_generator.generator import generate_fraud_subset
from data_generator.stress_scenarios import (apply_stress,
                                             get_available_scenarios)
from database import (create_experiment, finish_experiment, init_db,
                      save_model_results)

# =====================================================
# IMPORT GENERATOR
# =====================================================


# =====================================================
# IMPORT STRESS SCENARIOS
# =====================================================


# =====================================================
# LOGGING
# =====================================================

logging.basicConfig(
    filename="database/app.log",
    level=logging.ERROR,
    format="%(asctime)s %(levelname)s %(message)s",
)

# =====================================================
# DATA LOADING FUNCTIONS
# =====================================================


def load_or_generate_data(uploaded_file, sample_size, fraud_ratio):
    """Loads user CSV or generates synthetic data"""
    if uploaded_file is not None:
        if not uploaded_file.name.endswith(".csv"):
            st.error("Invalid file extension")
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
        with st.spinner("Generating synthetic data..."):
            df = generate_fraud_subset(
                subset_size=sample_size,
                full_size=2000,
                fraud_ratio=fraud_ratio,
                label_noise=0.015,
                random_state=42,
                use_stratification=True,
            )
        return df, "synthetic"


# =====================================================
# DATA PREPARATION FUNCTIONS FOR MODELS
# =====================================================

# 20 numeric features for LR/RF models
NUMERIC_FEATURES_20 = [
    "amount",
    "transaction_hour",
    "day_of_week",
    "distance_km",
    "nfc_time_exceeded",
    "nfc_duration_ms",
    "is_unusual_amount",
    "is_unusual_time",
    "sms_anomaly_6h",
    "phone_changed_48h",
    "device_risk_high",
    "device_age_days",
    "suspect_cash_deposit",
    "new_beneficiary_after_self_transfer",
    "age",
    "city_pop",
    "avg_amount_client",
    "std_amount_client",
    "typical_hour_client",
    "device_risk",
]


def prepare_data_for_lr_rf(X):
    """Prepare data for Logistic Regression and Random Forest (20 features)"""
    # Select only the 20 required features
    available_cols = [col for col in NUMERIC_FEATURES_20 if col in X.columns]
    X_result = X[available_cols].copy()

    # Add missing columns with 0
    for col in NUMERIC_FEATURES_20:
        if col not in X_result.columns:
            X_result[col] = 0

    # Ensure correct column order
    return X_result[NUMERIC_FEATURES_20]


def prepare_data_for_catboost(X):
    """Prepare data for CatBoost with correct column order"""
    X = X.copy()

    # ТОЧНЫЙ ПОРЯДОК КОЛОНОК, КАК В МОДЕЛИ (из отладки)
    expected_order = [
        "amount",
        "category",
        "transaction_hour",
        "day_of_week",
        "distance_km",
        "nfc_time_exceeded",
        "nfc_duration_ms",
        "is_unusual_amount",
        "is_unusual_time",
        "sms_anomaly_6h",
        "phone_changed_48h",
        "device_risk_high",
        "device_age_days",
        "suspect_cash_deposit",
        "new_beneficiary_after_self_transfer",
        "age",
        "city_pop",
        "avg_amount_client",
        "std_amount_client",
        "typical_hour_client",
        "device_risk",
    ]

    # Добавляем недостающие колонки
    for col in expected_order:
        if col not in X.columns:
            if col == "category":
                X[col] = "unknown"
            else:
                X[col] = 0.0

    # Приводим типы
    for col in expected_order:
        if col == "category":
            X[col] = X[col].astype(str)
        else:
            X[col] = X[col].astype(float)

    # Возвращаем в правильном порядке
    return X[expected_order]


def prepare_data_for_iforest(X, model):
    """Prepare data for Isolation Forest (one-hot encoding)"""
    X_processed = pd.get_dummies(X.copy())
    if hasattr(model, "feature_names_in_"):
        expected_cols = list(model.feature_names_in_)
        X_processed = X_processed.reindex(columns=expected_cols, fill_value=0)
    return X_processed


def prepare_model_data(model_name, model, X):
    """Universal function to prepare data for any model"""
    if model_name in ["Logistic Regression", "Random Forest v1", "Random Forest v2"]:
        return prepare_data_for_lr_rf(X)
    elif "CatBoost" in model_name:
        return prepare_data_for_catboost(X)
    elif model_name == "Isolation Forest":
        return prepare_data_for_iforest(X, model)
    return X.copy()


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
    models["CatBoost v1"] = joblib.load("models/advanced_models/catboost_v1.pkl")
    models["CatBoost v2"] = joblib.load("models/advanced_models/catboost_v2.pkl")
    models["Isolation Forest"] = joblib.load(
        "models/advanced_models/isolation_forest.pkl"
    )
    return models


# =====================================================
# BUSINESS COST CALCULATION
# =====================================================


def calculate_business_cost(y_true, y_pred, fp_weight, fn_weight):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return fp * fp_weight + fn * fn_weight


# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(page_title="Anti-Fraud System", layout="wide")
init_db()
st.title("Anti-Fraud System")

with st.expander("Stress scenarios information"):
    st.markdown(
        """
    | Scenario | Description | Purpose |
    |----------|------------|-------------|
    | `normal` | No changes | Baseline comparison |
    | `imbalance` | Fraud ratio drops to 0.1% | Test with rare fraud |
    | `amount_shift` | Fraud amounts decrease 10x | Test when fraudsters hide large amounts |
    | `masking` | Fraud features shift towards normal | Test when fraudsters try to look normal |
    | `frequency_boost` | All features increase 1.5x | Test activity spikes |
    """
    )

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.header("Settings")

sample_size = st.sidebar.slider(
    "Sample size", 100, 2000, 1000, 100, help="Number of transactions to analyze"
)

threshold = st.sidebar.slider(
    "Classification threshold",
    0.1,
    0.9,
    0.5,
    0.05,
    help="Probability above this threshold is considered fraud",
)

fraud_ratio = st.sidebar.slider(
    "Fraud ratio",
    0.01,
    0.30,
    0.05,
    0.01,
    format="%.2f",
    help="Fraud ratio in synthetic data",
)

st.sidebar.markdown("---")
st.sidebar.subheader("Error costs")

fp_weight = st.sidebar.slider(
    "False Positive cost", 1, 100, 1, help="Cost of blocking a legitimate transaction"
)
fn_weight = st.sidebar.slider(
    "False Negative cost", 1, 100, 10, help="Cost of missing a fraudulent transaction"
)

available_scenarios = get_available_scenarios()
selected_scenario = st.sidebar.selectbox(
    "Stress scenario", available_scenarios, help="Simulate abnormal fraud behavior"
)

all_models = [
    "Logistic Regression",
    "Random Forest v1",
    "Random Forest v2",
    "CatBoost v1",
    "CatBoost v2",
    "Isolation Forest",
]
selected_models = st.sidebar.multiselect(
    "Models to compare",
    options=all_models,
    default=all_models,
    help="Select models for evaluation and comparison",
)

uploaded_file = st.file_uploader(
    "Upload CSV",
    type=["csv"],
    help="File must contain columns matching the expected features",
)

compare_button = st.sidebar.button(
    "Run analysis", type="primary", help="Start model evaluation"
)

# =====================================================
# SESSION STATE INITIALIZATION
# =====================================================

defaults = {
    "classic_df_full": None,
    "X_classic": None,
    "y_classic": None,
    "stress_df_full": None,
    "X_stress": None,
    "y_stress": None,
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
# MAIN LOGIC - DATA LOADING/GENERATION
# =====================================================

if compare_button:
    st.toast(f"Starting analysis with scenario: {selected_scenario}")
    try:
        from metrics.validator import (get_column_info,
                                       prepare_data_for_prediction,
                                       validate_csv)

        classic_df_full, source = load_or_generate_data(
            uploaded_file, sample_size, fraud_ratio
        )
        if classic_df_full is None:
            st.stop()

        y_classic = (
            classic_df_full["is_fraud"].copy()
            if "is_fraud" in classic_df_full.columns
            else None
        )

        X_classic = prepare_data_for_prediction(classic_df_full)

        classic_df_display = X_classic.copy()
        if y_classic is not None:
            classic_df_display["is_fraud"] = y_classic.values

        st.session_state.classic_df_full = classic_df_display
        st.session_state.X_classic = X_classic
        st.session_state.y_classic = y_classic

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

        col_info = get_column_info()
        st.success(
            f"Data loaded! {len(X_classic)} rows, {len(X_classic.columns)} features."
        )

        with st.expander("Feature information"):
            st.markdown(f"**Total features:** {col_info['total_count']}")
            st.markdown("**Main features:**")
            for name, desc in list(col_info["sample_types"].items())[:5]:
                st.markdown(f"- `{name}`: {desc}")

    except Exception as e:
        logging.error(str(e))
        st.error(f"Error loading/generating data: {e}")

# =====================================================
# DATA DISPLAY AND EDITING
# =====================================================

if st.session_state.classic_df_full is not None:
    st.subheader("Dataset editing")
    edited_df = st.data_editor(
        st.session_state.classic_df_full, num_rows="dynamic", key="data_editor"
    )

    if not edited_df.equals(st.session_state.classic_df_full):
        from metrics.validator import prepare_data_for_prediction, validate_csv

        require_target = "is_fraud" in edited_df.columns
        is_valid, errors, warnings = validate_csv(
            edited_df, require_target=require_target
        )

        for w in warnings:
            st.warning(w)

        if is_valid:
            st.success("Edited data passed validation")

            st.session_state.classic_df_full = edited_df.copy()

            y_new = (
                edited_df["is_fraud"].copy()
                if "is_fraud" in edited_df.columns
                else None
            )
            X_new = prepare_data_for_prediction(edited_df)
            st.session_state.X_classic = X_new
            st.session_state.y_classic = y_new

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
            st.warning("Edited data has errors. Models will not run until fixed.")
            st.session_state.data_valid = False

    st.subheader("Generated data")
    st.dataframe(st.session_state.classic_df_full.head(), use_container_width=True)

    # =================================================
    # METRICS CALCULATION
    # =================================================
    models_changed = st.session_state.last_selected_models != selected_models
    threshold_changed = st.session_state.last_threshold != threshold
    need_recalc = (
        not st.session_state.results_calculated or models_changed or threshold_changed
    )

    if need_recalc and st.session_state.data_valid:
        X_classic = st.session_state.X_classic
        y_classic = st.session_state.y_classic
        X_stress = st.session_state.X_stress
        y_stress = st.session_state.y_stress

        if y_classic is None:
            st.error(
                "Missing 'is_fraud' column (target variable). "
                "For synthetic data this is a generation error. "
                "For user CSV - add 0/1 labels."
            )
            st.stop()

        if y_stress is None:
            st.error("Missing 'is_fraud' column in stress data.")
            st.stop()

        all_models_dict = load_all_models()
        models = {
            name: all_models_dict[name]
            for name in selected_models
            if name in all_models_dict
        }
        st.success(f"Loaded {len(models)} models")

        results_classic = []
        results_stress = []

        for name, model in models.items():
            st.info(f"Analyzing model: {name}")

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

            if name == "Isolation Forest":
                classic_pred = (classic_pred == -1).astype(int)
                stress_pred = (stress_pred == -1).astype(int)

            results_classic.append(
                {
                    "Model": name,
                    "Precision": round(
                        precision_score(y_classic, classic_pred, zero_division=0), 4
                    ),
                    "Recall": round(
                        recall_score(y_classic, classic_pred, zero_division=0), 4
                    ),
                    "F1": round(f1_score(y_classic, classic_pred, zero_division=0), 4),
                    "Business Cost": calculate_business_cost(
                        y_classic, classic_pred, fp_weight, fn_weight
                    ),
                }
            )
            results_stress.append(
                {
                    "Model": name,
                    "Precision": round(
                        precision_score(y_stress, stress_pred, zero_division=0), 4
                    ),
                    "Recall": round(
                        recall_score(y_stress, stress_pred, zero_division=0), 4
                    ),
                    "F1": round(f1_score(y_stress, stress_pred, zero_division=0), 4),
                    "Business Cost": calculate_business_cost(
                        y_stress, stress_pred, fp_weight, fn_weight
                    ),
                }
            )

        st.session_state.results_classic = pd.DataFrame(results_classic)
        st.session_state.results_stress = pd.DataFrame(results_stress)
        st.session_state.last_selected_models = selected_models.copy()
        st.session_state.last_threshold = threshold
        st.session_state.results_calculated = True

    # =================================================
    # DISPLAY RESULTS AND DATABASE LOGGING
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

        st.subheader("Classic mode")
        st.dataframe(df_classic, use_container_width=True, hide_index=True)

        st.subheader(f"Stress mode: {selected_scenario}")
        st.dataframe(df_stress, use_container_width=True, hide_index=True)

        st.subheader("F1 score comparison")
        chart_df = pd.DataFrame(
            {
                "Model": df_classic["Model"],
                "F1 Classic": df_classic["F1"].str.replace("%", "").astype(float),
                "F1 Stress": df_stress["F1"].str.replace("%", "").astype(float),
            }
        )
        fig = px.bar(
            chart_df, x="Model", y=["F1 Classic", "F1 Stress"], barmode="group"
        )
        st.plotly_chart(fig, use_container_width=True)

        # DATABASE LOGGING
        try:
            exp_id = create_experiment(
                sample_size=sample_size,
                threshold=threshold,
                fraud_ratio=fraud_ratio,
                stress_scenario=selected_scenario,
                models_used=selected_models,
            )
            for _, row in df_classic.iterrows():
                business_cost = row["Business Cost"]
                if pd.isna(business_cost) or business_cost is None:
                    business_cost = 0

                save_model_results(
                    exp_id=exp_id,
                    model_name=row["Model"],
                    mode="classic",
                    precision=float(row["Precision"].rstrip("%")) / 100,
                    recall=float(row["Recall"].rstrip("%")) / 100,
                    f1=float(row["F1"].rstrip("%")) / 100,
                    business_cost=business_cost,
                )
            for _, row in df_stress.iterrows():
                business_cost = row["Business Cost"]
                if pd.isna(business_cost) or business_cost is None:
                    business_cost = 0

                save_model_results(
                    exp_id=exp_id,
                    model_name=row["Model"],
                    mode="stress",
                    precision=float(row["Precision"].rstrip("%")) / 100,
                    recall=float(row["Recall"].rstrip("%")) / 100,
                    f1=float(row["F1"].rstrip("%")) / 100,
                    business_cost=business_cost,
                )
            finish_experiment(exp_id)
            st.success(f"Experiment saved to database (ID: {exp_id})")
        except Exception as db_err:
            logging.error(f"DB error: {db_err}")
            st.warning("Could not save results to database")

        st.success("Analysis completed")
