"""
Тесты для генератора данных
"""
import pytest
from data_generator.generator import generate_transactions
from data_generator.stress_scenarios import apply_stress

def test_generate_transactions_default():
    """Проверяем, что генератор создаёт 1000 строк с 5% фрода"""
    df = generate_transactions(1000)
    assert len(df) == 1000
    assert 'is_fraud' in df.columns
    assert df['is_fraud'].sum() == 50

def test_generate_transactions_custom_size():
    """Проверяем произвольный размер выборки"""
    df = generate_transactions(500)
    assert len(df) == 500

def test_generate_transactions_custom_fraud_rate():
    """Проверяем изменение доли фрода"""
    df = generate_transactions(200, fraud_prob=0.1)
    assert df['is_fraud'].sum() == 20

def test_generate_transactions_reproducibility():
    """Проверяем воспроизводимость с одинаковым random_state"""
    df1 = generate_transactions(100, random_state=42)
    df2 = generate_transactions(100, random_state=42)
    assert df1['is_fraud'].sum() == df2['is_fraud'].sum()

def test_stress_imbalance():
    """Проверяем стресс-сценарий imbalance"""
    df = generate_transactions(1000)
    df_stressed = apply_stress(df, 'imbalance')
    fraud_rate = df_stressed['is_fraud'].sum() / len(df_stressed)
    assert fraud_rate <= 0.01

def test_stress_amount_shift():
    """Проверяем стресс-сценарий amount_shift"""
    df = generate_transactions(100)
    df_stressed = apply_stress(df, 'amount_shift')
    fraud_mask = df_stressed['is_fraud'] == 1
    if fraud_mask.any():
        assert (df_stressed.loc[fraud_mask, 'feature_0'] != df.loc[fraud_mask, 'feature_0']).any()

def test_generate_fraud_subset_default():
    """Проверяем генерацию подвыборки по умолчанию"""
    df = generate_fraud_subset(subset_size=500)
    assert len(df) == 500
    assert 'is_fraud' in df.columns

def test_generate_fraud_subset_size_range():
    """Проверяем границы размера подвыборки"""
    df_min = generate_fraud_subset(subset_size=100)
    assert len(df_min) == 100
    df_max = generate_fraud_subset(subset_size=2000)
    assert len(df_max) == 2000

def test_generate_fraud_subset_invalid_size():
    """Проверяем, что недопустимый размер вызывает ошибку"""
    import pytest
    with pytest.raises(ValueError):
        generate_fraud_subset(subset_size=50)
    with pytest.raises(ValueError):
        generate_fraud_subset(subset_size=2500)

def test_generate_fraud_subset_reproducibility():
    """Проверяем воспроизводимость"""
    df1 = generate_fraud_subset(subset_size=300, random_state=42)
    df2 = generate_fraud_subset(subset_size=300, random_state=42)
    assert df1['is_fraud'].sum() == df2['is_fraud'].sum()
