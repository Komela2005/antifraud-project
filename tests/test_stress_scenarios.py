"""
Тесты для стресс-сценариев (адаптировано под новый генератор)
"""

from data_generator import generate_fraud_dataset
from data_generator.stress_scenarios import (apply_stress,
                                             get_available_scenarios)


def test_all_scenarios_work():
    """Проверяем, что все сценарии применяются без ошибок"""
    _, _, df = generate_fraud_dataset(n_transactions=500)
    scenarios = ["normal", "imbalance", "amount_shift", "masking", "frequency_boost"]

    for scenario in scenarios:
        df_stressed = apply_stress(df, scenario)
        assert len(df_stressed) == len(df)
        assert "is_fraud" in df_stressed.columns


def test_imbalance_reduces_fraud_rate():
    """Проверяем, что imbalance снижает долю фрода до ~0.1%"""
    _, _, df = generate_fraud_dataset(n_transactions=1000, fraud_ratio=0.05)
    df_stressed = apply_stress(df, "imbalance")

    fraud_rate = df_stressed["is_fraud"].sum() / len(df_stressed)
    assert fraud_rate <= 0.01


def test_amount_shift_modifies_amount():
    """Проверяем, что amount_shift изменяет сумму у мошенников"""
    _, _, df = generate_fraud_dataset(n_transactions=500, fraud_ratio=0.1)
    df_stressed = apply_stress(df, "amount_shift")

    fraud_mask = df["is_fraud"] == 1
    if fraud_mask.any():
        original_amounts = df.loc[fraud_mask, "amount"].values
        stressed_amounts = df_stressed.loc[fraud_mask, "amount"].values
        # Суммы должны уменьшиться (умножение на 0.1)
        assert (stressed_amounts <= original_amounts).all()


def test_frequency_boost_increases_values():
    """Проверяем, что frequency_boost увеличивает числовые признаки"""
    _, _, df = generate_fraud_dataset(n_transactions=500)
    df_stressed = apply_stress(df, "frequency_boost")

    # Проверяем, что amount увеличился
    assert (df_stressed["amount"] >= df["amount"]).all()


def test_get_available_scenarios():
    """Проверяем, что функция возвращает словарь со сценариями"""
    scenarios = get_available_scenarios()
    assert "imbalance" in scenarios
    assert "amount_shift" in scenarios
    assert "masking" in scenarios
    assert "frequency_boost" in scenarios
    assert "normal" in scenarios


def test_masking_does_not_break():
    """Проверяем, что маскировка не ломает структуру данных"""
    _, _, df = generate_fraud_dataset(n_transactions=300)
    df_stressed = apply_stress(df, "masking")

    # Проверяем, что все колонки сохранились
    assert list(df.columns) == list(df_stressed.columns)
    # Проверяем, что нет NaN
    assert not df_stressed.isnull().any().any()
