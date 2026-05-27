import os
import warnings

import joblib
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

print("=" * 70)
print("ОБУЧЕНИЕ РАСШИРЕННЫХ МОДЕЛЕЙ (CatBoost + Isolation Forest)")
print("=" * 70)

# 1. Загрузка данных
print("\n1. Загрузка датасета...")
df = pd.read_csv("data/fraud_transaction_dataset.csv")
print(f"   Форма данных: {df.shape}")
print(f"   Доля фрода: {df['is_fraud'].mean():.4f}")

# 2. Подготовка признаков
exclude_cols = ["client_id", "trans_date", "merch_lat", "merch_lon", "is_fraud"]
feature_cols = [col for col in df.columns if col not in exclude_cols]

categorical_features = []
for col in feature_cols:
    if df[col].dtype == "object" or df[col].dtype.name == "category":
        categorical_features.append(col)

print(f"\n   Категориальные признаки: {categorical_features}")

X = df[feature_cols].copy()
y = df["is_fraud"].values

print(f"   Всего признаков: {len(feature_cols)}")
print(f"   Из них категориальных: {len(categorical_features)}")

# 3. Разделение на train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("\n2. Разделение данных:")
print(f"   Train: {X_train.shape[0]} строк")
print(f"   Test: {X_test.shape[0]} строк")

# CatBoost v1
print("\n" + "=" * 70)
print("CatBoost v1 (100 итераций, глубина 6)")
print("=" * 70)

print("\nОбучение CatBoost v1...")
cb_v1 = CatBoostClassifier(
    iterations=100,
    depth=6,
    learning_rate=0.1,
    random_seed=42,
    verbose=False,
    class_weights=[1, 10],
    cat_features=categorical_features,
)
cb_v1.fit(X_train, y_train)
print("✅ CatBoost v1 обучен")

y_pred_v1 = cb_v1.predict(X_test)
y_proba_v1 = cb_v1.predict_proba(X_test)[:, 1]

print("\nРезультаты CatBoost v1 на тестовой выборке:")
print(f"  Accuracy:  {accuracy_score(y_test, y_pred_v1):.4f}")
print(f"  Precision: {precision_score(y_test, y_pred_v1):.4f}")
print(f"  Recall:    {recall_score(y_test, y_pred_v1):.4f}")
print(f"  F1:        {f1_score(y_test, y_pred_v1):.4f}")
print(f"  ROC-AUC:   {roc_auc_score(y_test, y_proba_v1):.4f}")

# CatBoost v2
print("\n" + "=" * 70)
print("CatBoost v2 (300 итераций, глубина 8)")
print("=" * 70)

print("\nОбучение CatBoost v2...")
cb_v2 = CatBoostClassifier(
    iterations=300,
    depth=8,
    learning_rate=0.05,
    random_seed=42,
    verbose=False,
    class_weights=[1, 10],
    cat_features=categorical_features,
)
cb_v2.fit(X_train, y_train)
print("✅ CatBoost v2 обучен")

y_pred_v2 = cb_v2.predict(X_test)
y_proba_v2 = cb_v2.predict_proba(X_test)[:, 1]

print("\nРезультаты CatBoost v2 на тестовой выборке:")
print(f"  Accuracy:  {accuracy_score(y_test, y_pred_v2):.4f}")
print(f"  Precision: {precision_score(y_test, y_pred_v2):.4f}")
print(f"  Recall:    {recall_score(y_test, y_pred_v2):.4f}")
print(f"  F1:        {f1_score(y_test, y_pred_v2):.4f}")
print(f"  ROC-AUC:   {roc_auc_score(y_test, y_proba_v2):.4f}")

# Isolation Forest
print("\n" + "=" * 70)
print("Isolation Forest (обнаружение аномалий)")
print("=" * 70)

print("\nПодготовка числовых данных для Isolation Forest...")
X_numeric = pd.get_dummies(X, columns=categorical_features, drop_first=True)
X_train_numeric, X_test_numeric, y_train_numeric, y_test_numeric = train_test_split(
    X_numeric, y, test_size=0.2, random_state=42, stratify=y
)

print(f"  Числовых признаков после one-hot encoding: {X_numeric.shape[1]}")

print("\nОбучение Isolation Forest...")
iso_forest = IsolationForest(n_estimators=100, contamination=0.025, random_state=42)

X_train_normal = X_train_numeric[y_train_numeric == 0]
print(f"  Обучение только на нормальных транзакциях: {X_train_normal.shape[0]} строк")

iso_forest.fit(X_train_normal)
print("✅ Isolation Forest обучен")

y_pred_iso = iso_forest.predict(X_test_numeric)
y_pred_iso = (y_pred_iso == -1).astype(int)

print("\nРезультаты Isolation Forest на тестовой выборке:")
print(f"  Accuracy:  {accuracy_score(y_test_numeric, y_pred_iso):.4f}")
print(f"  Precision: {precision_score(y_test_numeric, y_pred_iso):.4f}")
print(f"  Recall:    {recall_score(y_test_numeric, y_pred_iso):.4f}")
print(f"  F1:        {f1_score(y_test_numeric, y_pred_iso):.4f}")

# Сохранение
print("\n" + "=" * 70)
print("Сохранение моделей")
print("=" * 70)

os.makedirs("models/advanced_models", exist_ok=True)

joblib.dump(cb_v1, "models/advanced_models/catboost_v1.pkl")
joblib.dump(cb_v2, "models/advanced_models/catboost_v2.pkl")
joblib.dump(iso_forest, "models/advanced_models/isolation_forest.pkl")

print("✅ Модели сохранены в папку 'models/advanced_models/':")
print("   - catboost_v1.pkl")
print("   - catboost_v2.pkl")
print("   - isolation_forest.pkl")

print("\n✅ ОБУЧЕНИЕ ЗАВЕРШЕНО!")
