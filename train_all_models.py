"""Универсальный скрипт для обучения всех моделей."""

import pandas as pd
import numpy as np
from src.training_utils import (
    load_data, split_data, get_categorical_features,
    train_logistic_regression, train_random_forest,
    train_catboost, train_isolation_forest,
    prepare_for_isolation_forest,
    evaluate_model, save_model
)

print("=" * 70)
print("ОБУЧЕНИЕ ВСЕХ МОДЕЛЕЙ")
print("=" * 70)

# =====================================================
# 1. ЗАГРУЗКА ДАННЫХ
# =====================================================
X, y, df, feature_cols = load_data('data/fraud_transaction_dataset.csv')

# =====================================================
# 2. ПОДГОТОВКА ДАННЫХ
# =====================================================
categorical_features = get_categorical_features(X)
print(f"\nCategorical features: {categorical_features}")

# Разделение на train/test
X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)

print(f"\nData split:")
print(f"   Train: {X_train.shape[0]} rows")
print(f"   Test: {X_test.shape[0]} rows")
print(f"   Fraud in train: {y_train.mean():.4f}")
print(f"   Fraud in test: {y_test.mean():.4f}")

# =====================================================
# 3. ОБУЧЕНИЕ ЛОГИСТИЧЕСКОЙ РЕГРЕССИИ
# =====================================================
print("\n" + "=" * 70)
print("1. LOGISTIC REGRESSION")
print("=" * 70)

lr = train_logistic_regression(X_train, y_train)
save_model(lr, 'models/450k_models/logistic_regression_450k.pkl')
evaluate_model(lr, X_test, y_test, "Logistic Regression")

# =====================================================
# 4. ОБУЧЕНИЕ RANDOM FOREST v1
# =====================================================
print("\n" + "=" * 70)
print("2. RANDOM FOREST v1 (100 деревьев, глубина 10)")
print("=" * 70)

rf1 = train_random_forest(X_train, y_train, n_estimators=100, max_depth=10, min_samples_split=10)
save_model(rf1, 'models/450k_models/random_forest_v1_450k.pkl')
evaluate_model(rf1, X_test, y_test, "Random Forest v1")

# =====================================================
# 5. ОБУЧЕНИЕ RANDOM FOREST v2
# =====================================================
print("\n" + "=" * 70)
print("3. RANDOM FOREST v2 (200 деревьев, глубина 20)")
print("=" * 70)

rf2 = train_random_forest(X_train, y_train, n_estimators=200, max_depth=20, min_samples_split=5, min_samples_leaf=2)
save_model(rf2, 'models/450k_models/random_forest_v2_450k.pkl')
evaluate_model(rf2, X_test, y_test, "Random Forest v2")

# =====================================================
# 6. ОБУЧЕНИЕ CATBOOST v1
# =====================================================
print("\n" + "=" * 70)
print("4. CATBOOST v1 (100 итераций, глубина 6)")
print("=" * 70)

cb1 = train_catboost(X_train, y_train, categorical_features, iterations=100, depth=6, learning_rate=0.1)
save_model(cb1, 'models/advanced_models/catboost_v1.pkl')
evaluate_model(cb1, X_test, y_test, "CatBoost v1")

# =====================================================
# 7. ОБУЧЕНИЕ CATBOOST v2
# =====================================================
print("\n" + "=" * 70)
print("5. CATBOOST v2 (300 итераций, глубина 8)")
print("=" * 70)

cb2 = train_catboost(X_train, y_train, categorical_features, iterations=300, depth=8, learning_rate=0.05)
save_model(cb2, 'models/advanced_models/catboost_v2.pkl')
evaluate_model(cb2, X_test, y_test, "CatBoost v2")

# =====================================================
# 8. ОБУЧЕНИЕ ISOLATION FOREST
# =====================================================
print("\n" + "=" * 70)
print("6. ISOLATION FOREST")
print("=" * 70)

# Подготовка данных (one-hot encoding)
X_numeric = prepare_for_isolation_forest(X, categorical_features)
X_train_numeric, X_test_numeric, y_train_numeric, y_test_numeric = split_data(X_numeric, y, test_size=0.2)

# Обучаем только на нормальных транзакциях
X_train_normal = X_train_numeric[y_train_numeric == 0]
print(f"   Training on normal transactions only: {X_train_normal.shape[0]} rows")

iso = train_isolation_forest(X_train_normal)
save_model(iso, 'models/advanced_models/isolation_forest.pkl')

# Предсказания (аномалия = -1 -> 1)
y_pred_iso = iso.predict(X_test_numeric)
y_pred_iso = (y_pred_iso == -1).astype(int)

metrics = evaluate_model(iso, X_test_numeric, y_test_numeric, "Isolation Forest")

print("\n" + "=" * 70)
print("ОБУЧЕНИЕ ЗАВЕРШЕНО!")
print("=" * 70)
