"""
Пакет для генерации синтетических данных и стресс-сценариев
"""
from .generator import generate_transactions, generate_fraud_dataset, generate_fraud_subset
from .stress_scenarios import apply_stress

__all__ = [
    'generate_transactions',
    'generate_fraud_dataset', 
    'generate_fraud_subset',
    'apply_stress'
]
