"""
Пакет для генерации синтетических данных и стресс-сценариев
"""
from .generator import generate_fraud_dataset, generate_fraud_subset, get_expected_columns
from .stress_scenarios import apply_stress

__all__ = [
    'generate_fraud_dataset', 
    'generate_fraud_subset',
    'get_expected_columns',
    'apply_stress'
]
