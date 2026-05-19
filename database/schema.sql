-- Схема базы данных для логирования экспериментов

-- Таблица экспериментов
CREATE TABLE IF NOT EXISTS experiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_name TEXT NOT NULL,
    experiment_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    description TEXT,
    status TEXT DEFAULT 'running'
);

-- Таблица результатов моделей
CREATE TABLE IF NOT EXISTS model_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER NOT NULL,
    model_name TEXT NOT NULL,
    scenario TEXT NOT NULL,
    accuracy REAL,
    precision REAL,
    recall REAL,
    f1_score REAL,
    roc_auc REAL,
    business_cost REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (experiment_id) REFERENCES experiments(id) ON DELETE CASCADE
);

-- Таблица параметров эксперимента
CREATE TABLE IF NOT EXISTS experiment_params (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER NOT NULL,
    param_name TEXT NOT NULL,
    param_value TEXT NOT NULL,
    FOREIGN KEY (experiment_id) REFERENCES experiments(id) ON DELETE CASCADE
);

-- Индексы для быстрого поиска
CREATE INDEX IF NOT EXISTS idx_model_results_experiment ON model_results(experiment_id);
CREATE INDEX IF NOT EXISTS idx_model_results_model ON model_results(model_name);
CREATE INDEX IF NOT EXISTS idx_experiment_params_experiment ON experiment_params(experiment_id);
