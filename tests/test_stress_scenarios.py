"""
Тесты для стресс-сценариев
"""
import pytest
import pandas as pd
import numpy as np
from data_generator.generator import generate_transactions
from data_generator.stress_scenarios import apply_stress, get_available_scenarios

def test_all_scenarios_work():
    """Проверяем, что все сценарии применяются без ошибок"""
    df = generate_transactions(200)
    scenarios = ['normal', 'imbalance', 'amount_shift', 'masking', 'frequency_boost']
    
    for scenario in scenarios:
        df_stressed = apply_stress(df, scenario)
        assert len(df_stressed) == len(df)
        assert 'is_fraud' in df_stressed.columns

def test_imbalance_reduces_fraud_rate():
    """Проверяем, что imbalance снижает долю фрода до ~0.1%"""
    df = generate_transactions(1000, fraud_prob=0.05)
    df_stressed = apply_stress(df, 'imbalance')
    
    fraud_rate = df_stressed['is_fraud'].sum() / len(df_stressed)
    assert fraud_rate <= 0.01

def test_amount_shift_modifies_feature_0():
    """Проверяем, что amount_shift изменяет признак feature_0 у мошенников"""
    df = generate_transactions(500, fraud_prob=0.1)
    df_stressed = apply_stress(df, 'amount_shift')
    
    fraud_mask = df['is_fraud'] == 1
    if fraud_mask.any():
        original_values = df.loc[fraud_mask, 'feature_0'].values
        stressed_values = df_stressed.loc[fraud_mask, 'feature_0'].values
        assert not np.array_equal(original_values, stressed_values)

def test_get_available_scenarios():
    """Проверяем, что функция возвращает словарь со сценариями"""
    scenarios = get_available_scenarios()
    assert 'imbalance' in scenarios
    assert 'amount_shift' in scenarios
    assert 'masking' in scenarios
    assert 'frequency_boost' in scenarios
