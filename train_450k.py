import joblib
import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("ОБУЧЕНИЕ МОДЕЛЕЙ НА РЕАЛЬНОМ ДАТАСЕТЕ (504,883 строк)")
print("=" * 70)

# 1. Загружаем датасет
print("\n1. Загрузка датасета...")
df = pd.read_csv('data/fraud_transaction_dataset.csv')
print(f"   Форма: {df.shape}")
print(f"   Доля фрода: {df['is_fraud'].mean():.4f} ({df['is_fraud'].sum():.0f} из {len(df)})")

# 2. Отбираем признаки для обучения
# Исключаем нечисловые и служебные колонки
exclude_cols = ['client_id', 'trans_date', 'merch_lat', 'merch_lon', 'category', 'is_fraud']
feature_cols = [col for col in df.columns if col not in exclude_cols]

X = df[feature_cols].values
y = df['is_fraud'].values

print(f"\n2. Подготовка данных:")
print(f"   Признаков: {X.shape[1]}")
print(f"   Колонки-признаки: {feature_cols[:10]}...")

# 3. Разделяем на train/test для честной оценки
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n3. Разделение на train/test:")
print(f"   Train: {X_train.shape[0]} строк")
print(f"   Test: {X_test.shape[0]} строк")
print(f"   Доля фрода в train: {y_train.mean():.4f}")
print(f"   Доля фрода в test: {y_test.mean():.4f}")

# 4. Обучение моделей
print("\n" + "=" * 70)
print("ОБУЧЕНИЕ МОДЕЛЕЙ")
print("=" * 70)

# Logistic Regression
print("\n📊 Logistic Regression...")
lr = Pipeline([
    ('scaler', StandardScaler()),
    ('lr', LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced', n_jobs=-1))
])
lr.fit(X_train, y_train)
print("   ✅ Logistic Regression обучена")

# Random Forest v1
print("\n🌳 Random Forest v1 (100 деревьев, глубина 10)...")
rf1 = RandomForestClassifier(
    n_estimators=100, max_depth=10, min_samples_split=10,
    random_state=42, class_weight='balanced', n_jobs=-1
)
rf1.fit(X_train, y_train)
print("   ✅ Random Forest v1 обучен")

# Random Forest v2
print("\n🌲 Random Forest v2 (200 деревьев, глубина 20)...")
rf2 = RandomForestClassifier(
    n_estimators=200, max_depth=20, min_samples_split=5, min_samples_leaf=2,
    random_state=42, class_weight='balanced', n_jobs=-1
)
rf2.fit(X_train, y_train)
print("   ✅ Random Forest v2 обучен")

# 5. Сохраняем модели
print("\n💾 Сохранение моделей...")
os.makedirs('models', exist_ok=True)
joblib.dump(lr, 'models/logistic_regression_450k.pkl')
joblib.dump(rf1, 'models/random_forest_v1_450k.pkl')
joblib.dump(rf2, 'models/random_forest_v2_450k.pkl')
print("   ✅ Модели сохранены в папку 'models/'")

# 6. Оценка на тестовой выборке
print("\n" + "=" * 70)
print("РЕЗУЛЬТАТЫ НА ТЕСТОВОЙ ВЫБОРКЕ (20% данных)")
print("=" * 70)

for name, model in [('Logistic Regression', lr),
                    ('Random Forest v1', rf1),
                    ('Random Forest v2', rf2)]:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    print(f"\n📈 {name}:")
    print(f"   Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"   Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"   Recall:    {recall_score(y_test, y_pred):.4f}")
    print(f"   F1:        {f1_score(y_test, y_pred):.4f}")
    print(f"   ROC-AUC:   {roc_auc_score(y_test, y_proba):.4f}")

print("\n" + "=" * 70)
print("✅ ОБУЧЕНИЕ ЗАВЕРШЕНО!")
print("=" * 70)
print("\nМодели сохранены:")
print("   - models/logistic_regression_450k.pkl")
print("   - models/random_forest_v1_450k.pkl")
print("   - models/random_forest_v2_450k.pkl")
