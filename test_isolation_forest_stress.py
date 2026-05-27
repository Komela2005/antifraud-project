import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from data_generator.generator import generate_fraud_subset
from data_generator.stress_scenarios import apply_stress

print("=" * 80)
print("ПРОВЕРКА УСТОЙЧИВОСТИ ISOLATION FOREST К СТРЕСС-СЦЕНАРИЯМ")
print("=" * 80)

# Загрузка модели Isolation Forest
print("\n1. Загрузка модели Isolation Forest...")
iso_forest = joblib.load('models/advanced_models/isolation_forest.pkl')
print("   Модель загружена")

# Список стресс-сценариев
scenarios = ['normal', 'imbalance', 'amount_shift', 'masking']
results = []

# Получаем ожидаемые признаки модели
expected_features = iso_forest.feature_names_in_
print(f"\n   Модель ожидает {len(expected_features)} признаков")

print("\n2. Тестирование на разных стресс-сценариях:")
print("-" * 80)

for scenario in scenarios:
    print(f"\nСценарий: {scenario}")
    
    # Генерация тестовых данных
    df = generate_fraud_subset(
        subset_size=2000,
        full_size=2000,
        fraud_ratio=0.05,
        random_state=42
    )
    
    # Применение стресс-сценария
    if scenario != 'normal':
        df_stressed = apply_stress(df.copy(), scenario)
    else:
        df_stressed = df.copy()
    
    # Подготовка признаков (все, кроме is_fraud)
    feature_cols = [col for col in df_stressed.columns if col != 'is_fraud']
    
    # Проверяем, какие колонки есть
    print(f"   Доступно признаков: {len(feature_cols)}")
    
    X_test = df_stressed[feature_cols]
    y_true = df_stressed['is_fraud'].values
    
    # Для Isolation Forest нужно преобразовать данные так же, как при обучении
    # Обучался на 27 признаках. Создадим DataFrame с нужными колонками
    X_test_aligned = pd.DataFrame(index=X_test.index)
    
    for col in expected_features:
        if col in X_test.columns:
            X_test_aligned[col] = X_test[col]
        else:
            # Если колонки нет, заполняем 0
            X_test_aligned[col] = 0
    
    print(f"   После выравнивания: {X_test_aligned.shape[1]} признаков")
    
    # Предсказание Isolation Forest
    y_pred_raw = iso_forest.predict(X_test_aligned)
    y_pred = (y_pred_raw == -1).astype(int)
    
    # Расчёт метрик
    metrics = {
        'scenario': scenario,
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'n_fraud_true': y_true.sum(),
        'n_fraud_pred': y_pred.sum(),
        'n_correct_fraud': ((y_true == 1) & (y_pred == 1)).sum()
    }
    results.append(metrics)
    
    print(f"   Accuracy:  {metrics['accuracy']:.4f}")
    print(f"   Precision: {metrics['precision']:.4f}")
    print(f"   Recall:    {metrics['recall']:.4f}")
    print(f"   F1:        {metrics['f1']:.4f}")
    print(f"   Найдено фрода: {metrics['n_correct_fraud']} из {metrics['n_fraud_true']}")

# Сводная таблица
print("\n" + "=" * 80)
print("СВОДНАЯ ТАБЛИЦА УСТОЙЧИВОСТИ ISOLATION FOREST")
print("=" * 80)

df_results = pd.DataFrame(results)
print(df_results.to_string(index=False))

# Расчёт просадки
print("\n" + "=" * 80)
print("ПРОСАДКА МЕТРИК ОТНОСИТЕЛЬНО NORMAL")
print("=" * 80)

normal_metrics = df_results[df_results['scenario'] == 'normal'].iloc[0]

for _, row in df_results.iterrows():
    scenario = row['scenario']
    if scenario != 'normal':
        drop_accuracy = (normal_metrics['accuracy'] - row['accuracy']) / normal_metrics['accuracy'] * 100 if normal_metrics['accuracy'] > 0 else 0
        drop_precision = (normal_metrics['precision'] - row['precision']) / normal_metrics['precision'] * 100 if normal_metrics['precision'] > 0 else 0
        drop_recall = (normal_metrics['recall'] - row['recall']) / normal_metrics['recall'] * 100 if normal_metrics['recall'] > 0 else 0
        drop_f1 = (normal_metrics['f1'] - row['f1']) / normal_metrics['f1'] * 100 if normal_metrics['f1'] > 0 else 0
        
        print(f"\n{scenario}:")
        print(f"   Accuracy:  {drop_accuracy:+.1f}%")
        print(f"   Precision: {drop_precision:+.1f}%")
        print(f"   Recall:    {drop_recall:+.1f}%")
        print(f"   F1:        {drop_f1:+.1f}%")

# Вывод
print("\n" + "=" * 80)
print("ВЫВОДЫ")
print("=" * 80)

best_f1 = df_results.loc[df_results['f1'].idxmax()]
worst_f1 = df_results.loc[df_results['f1'].idxmin()]

print(f"\nЛучший F1: {best_f1['scenario']} ({best_f1['f1']:.4f})")
print(f"Худший F1: {worst_f1['scenario']} ({worst_f1['f1']:.4f})")

if worst_f1['f1'] < best_f1['f1'] * 0.5:
    print("\nIsolation Forest НЕ устойчив к стресс-сценариям (F1 упал более чем на 50%)")
else:
    print("\nIsolation Forest устойчив к стресс-сценариям")

# Сохранение результатов
df_results.to_csv('isolation_forest_stress_results.csv', index=False)
print("\nРезультаты сохранены в isolation_forest_stress_results.csv")
