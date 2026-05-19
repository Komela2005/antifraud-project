"""
Модуль для логирования экспериментов в БД
Использует database модуль от Back1
"""
import sys
from pathlib import Path

# Добавляем путь к корневой папке проекта (чтобы импортировать модуль database)
sys.path.insert(0, str(Path(__file__).parent.parent))

from database import (
    create_experiment,
    save_model_results,
    save_experiment_params,
    finish_experiment
)


def log_experiment(
    experiment_name: str,
    params: dict,
    all_results: dict,
    description: str = None
) -> int:
    """
    Сохраняет результаты стресс-тестирования в БД
    
    Параметры:
    - experiment_name: название эксперимента
    - params: параметры эксперимента (sample_size, cost_fp, cost_fn и т.д.)
    - all_results: результаты evaluate_all_models()
    - description: описание (опционально)
    
    Возвращает:
    - experiment_id: ID созданного эксперимента
    """
    # 1. Создаём эксперимент
    exp_id = create_experiment(experiment_name, description)
    
    # 2. Сохраняем параметры
    save_experiment_params(exp_id, params)
    
    # 3. Сохраняем результаты для каждой модели и сценария
    for model_name, scenarios in all_results.items():
        for scenario, metrics in scenarios.items():
            save_model_results(
                experiment_id=exp_id,
                model_name=model_name,
                scenario=scenario,
                metrics={
                    'accuracy': metrics.get('Accuracy'),
                    'precision': metrics.get('Precision'),
                    'recall': metrics.get('Recall'),
                    'f1_score': metrics.get('F1'),
                    'roc_auc': metrics.get('ROC_AUC'),
                    'business_cost': metrics.get('Business_Cost')
                }
            )
    
    # 4. Завершаем эксперимент
    finish_experiment(exp_id)
    
    print(f"Эксперимент сохранён в БД (ID: {exp_id})")
    return exp_id


