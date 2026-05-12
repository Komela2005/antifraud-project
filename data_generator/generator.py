import numpy as np
import pandas as pd

def generate_transactions(n_samples=1000, fraud_prob=0.05, random_state=42):
    """
    Генерирует синтетический датасет транзакций
    
    Parameters:
    n_samples: int - количество транзакций (100-5000)
    fraud_prob: float - доля мошеннических транзакций (0.001-0.3)
    random_state: int - seed для воспроизводимости
    
    Returns:
    pd.DataFrame с 10 признаками и колонкой 'is_fraud'
    """
    np.random.seed(random_state)
    
    n_fraud = int(n_samples * fraud_prob)
    n_normal = n_samples - n_fraud
    
    # Нормальные транзакции
    normal_features = np.random.normal(loc=0, scale=1, size=(n_normal, 10))
    
    # Мошеннические транзакции (смещённые распределения)
    fraud_features = np.random.normal(loc=1.5, scale=1.2, size=(n_fraud, 10))
    
    # Объединяем
    X = np.vstack([normal_features, fraud_features])
    y = np.array([0] * n_normal + [1] * n_fraud)
    
    # Перемешиваем
    idx = np.random.permutation(n_samples)
    X, y = X[idx], y[idx]
    
    # Создаём DataFrame
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
    df['is_fraud'] = y
    
    return df

def get_expected_columns():
    """Возвращает список ожидаемых колонок для валидации CSV"""
    return [f'feature_{i}' for i in range(10)]

def generate_custom_fraud_distribution(n_samples=1000, fraud_mode='default'):
    """Генерация с разными типами распределения фрода"""
    if fraud_mode == 'rare':
        fraud_prob = 0.001
    elif fraud_mode == 'frequent':
        fraud_prob = 0.3
    else:
        fraud_prob = 0.05
    return generate_transactions(n_samples, fraud_prob)
