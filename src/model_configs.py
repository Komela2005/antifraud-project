"""Конфигурации гиперпараметров для моделей."""

LR_CONFIG = {
    'random_state': 42,
    'max_iter': 1000,
    'class_weight': 'balanced'
}

RF_V1_CONFIG = {
    'n_estimators': 100,
    'max_depth': 10,
    'min_samples_split': 10,
    'random_state': 42,
    'class_weight': 'balanced',
    'n_jobs': -1
}

RF_V2_CONFIG = {
    'n_estimators': 200,
    'max_depth': 20,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'random_state': 42,
    'class_weight': 'balanced',
    'n_jobs': -1
}

CATBOOST_V1_CONFIG = {
    'iterations': 100,
    'depth': 6,
    'learning_rate': 0.1,
    'random_seed': 42,
    'verbose': False,
    'class_weights': [1, 10]
}

CATBOOST_V2_CONFIG = {
    'iterations': 300,
    'depth': 8,
    'learning_rate': 0.05,
    'random_seed': 42,
    'verbose': False,
    'class_weights': [1, 10]
}

ISOLATION_FOREST_CONFIG = {
    'n_estimators': 100,
    'contamination': 0.025,
    'random_state': 42
}
