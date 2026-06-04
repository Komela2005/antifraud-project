"""Универсальный скрипт для обучения всех моделей (с конфигами)."""

import pandas as pd
import numpy as np
from src.training_utils import (
    load_data, split_data, get_categorical_features,
    train_logistic_regression, train_random_forest,
    train_catboost, train_isolation_forest,
    prepare_for_isolation_forest, evaluate_model, save_model
)
from src.model_configs import (
    LR_CONFIG, RF_V1_CONFIG, RF_V2_CONFIG,
    CATBOOST_V1_CONFIG, CATBOOST_V2_CONFIG,
    ISOLATION_FOREST_CONFIG
)

print("=" * 70)
print("ОБУЧЕНИЕ ВСЕХ МОДЕЛЕЙ")
print("=" * 70)

# Загрузка данных
X, y, df, feature_cols = load_data('data/fraud_transaction_dataset.csv')
categorical_features = get_categorical_features(X)

# Разделение данных
X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)

print(f"\nTrain size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
print(f"Fraud in train: {y_train.mean():.4f}, Fraud in test: {y_test.mean():.4f}")

# Список моделей для обучения
models_config = [
    ("Logistic Regression", train_logistic_regression, LR_CONFIG, 'models/450k_models/logistic_regression_450k.pkl'),
    ("Random Forest v1", train_random_forest, RF_V1_CONFIG, 'models/450k_models/random_forest_v1_450k.pkl'),
    ("Random Forest v2", train_random_forest, RF_V2_CONFIG, 'models/450k_models/random_forest_v2_450k.pkl'),
    ("CatBoost v1", lambda X, y, **kw: train_catboost(X, y, categorical_features, **kw), CATBOOST_V1_CONFIG, 'models/advanced_models/catboost_v1.pkl'),
    ("CatBoost v2", lambda X, y, **kw: train_catboost(X, y, categorical_features, **kw), CATBOOST_V2_CONFIG, 'models/advanced_models/catboost_v2.pkl'),
]

# Обучение всех моделей
for name, train_func, config, save_path in models_config:
    print("\n" + "=" * 70)
    print(f"Training: {name}")
    print("=" * 70)
    
    model = train_func(X_train, y_train, **config)
    save_model(model, save_path)
    evaluate_model(model, X_test, y_test, name)

# Isolation Forest (отдельно)
print("\n" + "=" * 70)
print("Training: Isolation Forest")
print("=" * 70)

X_numeric = prepare_for_isolation_forest(X, categorical_features)
X_train_numeric, X_test_numeric, y_train_numeric, y_test_numeric = split_data(X_numeric, y, test_size=0.2)

X_train_normal = X_train_numeric[y_train_numeric == 0]
print(f"Training on normal transactions: {X_train_normal.shape[0]} rows")

iso = train_isolation_forest(X_train_normal, **ISOLATION_FOREST_CONFIG)
save_model(iso, 'models/advanced_models/isolation_forest.pkl')

y_pred_iso = iso.predict(X_test_numeric)
y_pred_iso = (y_pred_iso == -1).astype(int)
evaluate_model(iso, X_test_numeric, y_test_numeric, "Isolation Forest")

print("\n" + "=" * 70)
print("ВСЕ МОДЕЛИ ОБУЧЕНЫ!")
print("=" * 70)
