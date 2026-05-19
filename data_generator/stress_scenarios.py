import numpy as np
import pandas as pd

def apply_stress(df, scenario):
    """
    Применяет стресс-сценарий к данным (адаптировано под новый генератор)
    
    Сценарии:
    - 'normal': без изменений
    - 'imbalance': экстремальный дисбаланс (0.1% фрода)
    - 'amount_shift': сдвиг суммы у мошенников (уменьшение amount)
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
        # Уменьшаем сумму у мошенников (имитация сокрытия)
        df_copy = df.copy()
        fraud_mask = df_copy['is_fraud'] == 1
        df_copy.loc[fraud_mask, 'amount'] = df_copy.loc[fraud_mask, 'amount'] * 0.1
        return df_copy
    
    elif scenario == 'masking':
        # Маскировка: у мошенников признаки смещаются к норме
        df_copy = df.copy()
        fraud_mask = df_copy['is_fraud'] == 1
        
        # Для числовых признаков уменьшаем отклонение
        numeric_cols = ['amount', 'distance_km', 'nfc_duration_ms', 'time_deviation_min']
        for col in numeric_cols:
            if col in df_copy.columns:
                df_copy.loc[fraud_mask, col] = df_copy.loc[fraud_mask, col] * 0.5
        
        # Для бинарных признаков снижаем аномальность
        binary_cols = ['is_unusual_amount', 'is_unusual_time', 'nfc_time_exceeded', 
                       'sms_anomaly_6h', 'phone_changed_48h', 'suspect_cash_deposit',
                       'new_beneficiary_after_self_transfer', 'device_risk_high']
        for col in binary_cols:
            if col in df_copy.columns:
                # С вероятностью 50% сбрасываем аномалию
                mask_reset = fraud_mask & (np.random.rand(len(df_copy)) < 0.5)
                df_copy.loc[mask_reset, col] = 0
        return df_copy
    
    elif scenario == 'frequency_boost':
        # Увеличиваем все признаки (имитация всплеска активности)
        df_copy = df.copy()
        numeric_cols = ['amount', 'distance_km', 'nfc_duration_ms', 'time_deviation_min']
        for col in numeric_cols:
            if col in df_copy.columns:
                df_copy[col] = df_copy[col] * 1.5
        return df_copy
    
    else:
        raise ValueError(f"Неизвестный сценарий: {scenario}")


def get_available_scenarios():
    """Возвращает список доступных стресс-сценариев для UI"""
    return {
        'normal': 'Классический режим (без стресса)',
        'imbalance': 'Экстремальный дисбаланс (0.1% фрода)',
        'amount_shift': 'Сдвиг суммы у мошенников',
        'masking': 'Маскировка под нормальное поведение',
        'frequency_boost': 'Всплеск частоты операций'
    }
