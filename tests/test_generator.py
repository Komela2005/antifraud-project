import pytest

from data_generator import (generate_fraud_dataset, generate_fraud_subset,
                            get_expected_columns)


def test_generate_fraud_dataset():
    X, y, df = generate_fraud_dataset(n_transactions=500)
    assert len(df) == 500
    assert "is_fraud" in df.columns


def test_generate_fraud_subset_default():
    df = generate_fraud_subset(subset_size=500)
    assert len(df) == 500


def test_generate_fraud_subset_size_range():
    df_min = generate_fraud_subset(subset_size=100)
    assert len(df_min) == 100
    df_max = generate_fraud_subset(subset_size=2000)
    assert len(df_max) == 2000


def test_generate_fraud_subset_invalid_size():
    with pytest.raises(ValueError):
        generate_fraud_subset(subset_size=50)
    with pytest.raises(ValueError):
        generate_fraud_subset(subset_size=2500)


def test_get_expected_columns():
    cols = get_expected_columns()
    assert isinstance(cols, list)
    assert len(cols) > 0
