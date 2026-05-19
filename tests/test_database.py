"""
Тесты для модуля работы с базой данных
"""
import pytest
import pandas as pd
from database.db_manager import (
    init_db, create_experiment, save_model_results,
    save_experiment_params, finish_experiment,
    get_experiment_results, get_all_experiments,
    get_experiment_summary, delete_experiment
)

@pytest.fixture
def setup_db():
    """Создаёт тестовую БД перед каждым тестом"""
    init_db()
    yield
    # Очистка после тестов
    delete_experiment(1)

def test_create_experiment():
    """Проверяет создание эксперимента"""
    exp_id = create_experiment("Тест", "Описание")
    assert exp_id == 1

def test_save_model_results():
    """Проверяет сохранение результатов модели"""
    exp_id = create_experiment("Тест моделей", "Проверка сохранения")
    
    save_model_results(
        experiment_id=exp_id,
        model_name="Test Model",
        scenario="normal",
        metrics={'accuracy': 0.95, 'precision': 0.90, 'recall': 0.88,
                 'f1_score': 0.89, 'roc_auc': 0.96, 'business_cost': 100}
    )
    
    results = get_experiment_results(exp_id)
    assert len(results) == 1
    assert results.iloc[0]['model_name'] == "Test Model"

def test_save_experiment_params():
    """Проверяет сохранение параметров эксперимента"""
    exp_id = create_experiment("Тест параметров")
    
    params = {'sample_size': 500, 'fraud_ratio': 0.01, 'scenario': 'imbalance'}
    save_experiment_params(exp_id, params)

def test_finish_experiment():
    """Проверяет изменение статуса эксперимента"""
    exp_id = create_experiment("Тест статуса")
    finish_experiment(exp_id, 'completed')
    
    summary = get_experiment_summary(exp_id)
    assert summary['status'] == 'completed'

def test_get_all_experiments():
    """Проверяет получение списка экспериментов"""
    create_experiment("Эксперимент 1")
    create_experiment("Эксперимент 2")
    
    experiments = get_all_experiments()
    assert len(experiments) >= 2

def test_delete_experiment():
    """Проверяет удаление эксперимента"""
    exp_id = create_experiment("Для удаления")
    delete_experiment(exp_id)
    
    experiments = get_all_experiments()
    assert experiments[experiments['id'] == exp_id].empty
