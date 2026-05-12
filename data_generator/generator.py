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

def generate_fraud_subset(
    subset_size: int = 500,
    full_size: int = 2000,
    fraud_ratio: float = 0.01,
    label_noise: float = 0.015,
    random_state: int = 42,
    use_stratification: bool = True
) -> pd.DataFrame:
    """
    Генерирует подвыборку указанного размера (100-2000) из основного генератора.
    """
    if not 100 <= subset_size <= 2000:
        raise ValueError(f"subset_size должен быть от 100 до 2000, получено {subset_size}")
    
    if subset_size > full_size:
        raise ValueError(f"subset_size ({subset_size}) не может быть больше full_size ({full_size})")
    
    _, _, full_df = generate_fraud_dataset(
        n_transactions=full_size,
        fraud_ratio=fraud_ratio,
        label_noise=label_noise,
        random_state=random_state
    )
    
    if use_stratification:
        n_fraud = int(subset_size * fraud_ratio)
        n_legit = subset_size - n_fraud
        
        fraud_df = full_df[full_df['is_fraud'] == 1]
        legit_df = full_df[full_df['is_fraud'] == 0]
        
        actual_n_fraud = min(n_fraud, len(fraud_df))
        actual_n_legit = min(n_legit, len(legit_df))
        
        if len(fraud_df) < n_fraud:
            actual_n_legit += (n_fraud - len(fraud_df))
        if len(legit_df) < n_legit:
            actual_n_fraud += (n_legit - len(legit_df))
        
        fraud_sample = fraud_df.sample(n=actual_n_fraud, random_state=random_state)
        legit_sample = legit_df.sample(n=actual_n_legit, random_state=random_state)
        
        subset_df = pd.concat([fraud_sample, legit_sample])
        subset_df = subset_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    else:
        subset_df = full_df.sample(n=subset_size, random_state=random_state).reset_index(drop=True)
    
    return subset_df
