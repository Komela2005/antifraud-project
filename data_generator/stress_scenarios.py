# =====================================================
# СТРЕСС-СЦЕНАРИИ ДЛЯ ТЕСТИРОВАНИЯ МОДЕЛЕЙ
# =====================================================

import numpy as np
import pandas as pd

# Константы
NUMERIC_COLS = ['amount', 'distance_km', 'nfc_duration_ms', 'time_deviation_min']


def apply_stress(df: pd.DataFrame, scenario: str) -> pd.DataFrame:
    """
    Применяет стресс-сценарий к данным.

    Parameters
    ----------
    df : pd.DataFrame
        Исходный датасет (должен содержать колонку 'is_fraud')
    scenario : str
        Название сценария:
        - 'normal' : без изменений
        - 'imbalance' : экстремальный дисбаланс (0.1% фрода)
        - 'amount_shift' : сдвиг суммы у мошенников
        - 'masking' : маскировка под нормальное поведение
        - 'frequency_boost' : всплеск частоты операций

    Returns
    -------
    pd.DataFrame
        Изменённый датасет

    Raises
    ------
    ValueError
        Если сценарий неизвестен
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
        # Сдвиг суммы у мошенников (уменьшение в 10 раз)
        df_copy = df.copy()
        fraud_mask = df_copy['is_fraud'] == 1
        df_copy.loc[fraud_mask, 'amount'] = df_copy.loc[fraud_mask, 'amount'] * 0.1
        return df_copy

    elif scenario == 'masking':
        # Маскировка под нормальное поведение
        df_copy = df.copy()
        fraud_mask = df_copy['is_fraud'] == 1
        for col in NUMERIC_COLS:
            if col in df_copy.columns:
                df_copy.loc[fraud_mask, col] = df_copy.loc[fraud_mask, col] * 0.5
        return df_copy

    elif scenario == 'frequency_boost':
        # Всплеск частоты операций
        df_copy = df.copy()
        for col in NUMERIC_COLS:
            if col in df_copy.columns:
                df_copy[col] = df_copy[col] * 1.5
        return df_copy

    else:
        raise ValueError(f"Неизвестный сценарий: {scenario}")


def get_available_scenarios() -> dict:
    """
    Возвращает список доступных стресс-сценариев для UI.

    Returns
    -------
    dict
        Словарь {ключ: описание}
    """
    return {
        'normal': 'Классический режим (без стресса)',
        'imbalance': 'Экстремальный дисбаланс (0.1% фрода)',
        'amount_shift': 'Сдвиг суммы у мошенников',
        'masking': 'Маскировка под нормальное поведение',
        'frequency_boost': 'Всплеск частоты операций'
    }