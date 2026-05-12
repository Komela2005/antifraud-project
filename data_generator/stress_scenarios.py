import numpy as np
import pandas as pd

def apply_stress(df, scenario):
    """
    Применяет стресс-сценарий к данным
    
    Сценарии:
    - 'normal': без изменений
    - 'imbalance': экстремальный дисбаланс (0.1% фрода)
    - 'amount_shift': сдвиг суммы у мошенников
    - 'masking': маскировка под нормальное поведение
    - 'frequency_boost': всплеск частоты операций
    """
    if scenario == 'normal':
        return df.copy()
    
    elif scenario == 'imbalance':
        # Уменьшаем долю фрода до 0.1%
        fraud_indices = df[df['is_fraud'] == 1].index
        n_fraud_to_keep = max(1, int(len(df) * 0.001))
        fraud_to_keep = np.random.choice(fraud_indices, n_fraud_to_keep, replace=False)
        df_copy = df.copy()
        df_copy.loc[df_copy.index.difference(fraud_to_keep), 'is_fraud'] = 0
        return df_copy
    
    elif scenario == 'amount_shift':
        # Сдвиг признака feature_0 у мошенников
        df_copy = df.copy()
        fraud_mask = df_copy['is_fraud'] == 1
        df_copy.loc[fraud_mask, 'feature_0'] = df_copy.loc[fraud_mask, 'feature_0'] * 0.1
        return df_copy
    
    elif scenario == 'masking':
        # Маскировка мошенников под норму
        df_copy = df.copy()
        fraud_mask = df_copy['is_fraud'] == 1
        for col in [f'feature_{i}' for i in range(10)]:
            df_copy.loc[fraud_mask, col] = df_copy.loc[fraud_mask, col] * 0.5
        return df_copy
    
    elif scenario == 'frequency_boost':
        # Всплеск активности
        df_copy = df.copy()
        df_copy[[f'feature_{i}' for i in range(10)]] *= 1.5
        return df_copy
    
    else:
        raise ValueError(f"Неизвестный сценарий: {scenario}")

def get_available_scenarios():
    """Возвращает список доступных стресс-сценариев"""
    return {
        'imbalance': 'Экстремальный дисбаланс (0.1% фрода)',
        'amount_shift': 'Сдвиг суммы у мошенников',
        'masking': 'Маскировка под нормальное поведение',
        'frequency_boost': 'Всплеск частоты операций'
    }
