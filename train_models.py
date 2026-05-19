import joblib
import sys
import os

# Добавляем путь к data_generator
sys.path.append('.')

# Импортируем из вашего генератора
from data_generator.generator import generate_transactions

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

print("=" * 50)
print("Обучение моделей для антифрод платформы")
print("=" * 50)

# 1. Генерируем 10k строк
print("\n1. Генерация синтетических данных (10,000 транзакций)...")
df = generate_transactions(
    n_samples=10000,
    fraud_prob=0.05,
    random_state=42
)

# Подготавливаем X и y
feature_cols = [f'feature_{i}' for i in range(10)]
X = df[feature_cols].values
y = df['is_fraud'].values

print(f"   Размер выборки: {X.shape[0]} строк, {X.shape[1]} признаков")
print(f"   Доля фрода: {y.mean():.3f} ({y.sum()} из {len(y)})")

# 2. Обучаем Logistic Regression
print("\n2. Обучение Logistic Regression...")
lr_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('lr', LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'))
])
lr_pipeline.fit(X, y)
print("   ✅ Logistic Regression обучена")

# 3. Обучаем Random Forest v1
print("\n3. Обучение Random Forest v1 (100 деревьев, глубина 10)...")
rf_v1 = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=10,
    random_state=42,
    class_weight='balanced',
    n_jobs=-1
)
rf_v1.fit(X, y)
print("   ✅ Random Forest v1 обучен")

# 4. Обучаем Random Forest v2
print("\n4. Обучение Random Forest v2 (200 деревьев, глубина 20)...")
rf_v2 = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    class_weight='balanced',
    n_jobs=-1
)
rf_v2.fit(X, y)
print("   ✅ Random Forest v2 обучен")

# 5. Сохраняем модели
print("\n5. Сохранение моделей...")
os.makedirs('models', exist_ok=True)

joblib.dump(lr_pipeline, 'models/logistic_regression.pkl')
joblib.dump(rf_v1, 'models/random_forest_v1.pkl')
joblib.dump(rf_v2, 'models/random_forest_v2.pkl')
print("   ✅ Модели сохранены в папку 'models/'")

# 6. Быстрая проверка качества
print("\n" + "=" * 50)
print("РЕЗУЛЬТАТЫ НА ОБУЧАЮЩЕЙ ВЫБОРКЕ (для справки)")
print("=" * 50)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

for name, model in [('Logistic Regression', lr_pipeline),
                    ('Random Forest v1', rf_v1),
                    ('Random Forest v2', rf_v2)]:
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    
    print(f"\n{name}:")
    print(f"  Accuracy:  {accuracy_score(y, y_pred):.4f}")
    print(f"  Precision: {precision_score(y, y_pred):.4f}")
    print(f"  Recall:    {recall_score(y, y_pred):.4f}")
    print(f"  F1:        {f1_score(y, y_pred):.4f}")
    print(f"  ROC-AUC:   {roc_auc_score(y, y_proba):.4f}")

print("\n" + "=" * 50)
print("✅ ГОТОВО! Модели можно использовать в приложении")
print("=" * 50)
