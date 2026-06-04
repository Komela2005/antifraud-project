"""
Модуль для работы с SQLite базой данных логирования экспериментов
"""

import sqlite3
from pathlib import Path

import pandas as pd

# Путь к файлу базы данных
DB_PATH = Path(__file__).parent / "experiments.db"


def get_connection():
    """Возвращает соединение с базой данных"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Инициализирует базу данных: создаёт таблицы, если их нет"""
    schema_path = Path(__file__).parent / "schema.sql"

    with open(schema_path, "r", encoding="utf-8") as f:
        schema_sql = f.read()

    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.executescript(schema_sql)
        conn.commit()

    print("База данных инициализирована")


def create_experiment(
    experiment_name: str = None,
    description: str = None,
    sample_size: int = None,
    threshold: float = None,
    fraud_ratio: float = None,
    stress_scenario: str = None,
    models_used: list = None,
) -> int:
    """
    Создаёт новый эксперимент

    Parameters:
    -----------
    experiment_name : str, optional
        Название эксперимента
    description : str, optional
        Описание эксперимента
    sample_size : int, optional
        Размер выборки
    threshold : float, optional
        Порог классификации
    fraud_ratio : float, optional
        Доля мошеннических транзакций
    stress_scenario : str, optional
        Стресс-сценарий
    models_used : list, optional
        Список использованных моделей

    Returns:
    --------
    int : ID созданного эксперимента
    """
    from datetime import datetime

    # Формируем название, если не указано
    if experiment_name is None:
        experiment_name = f"Experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Формируем описание из параметров
    if description is None:
        description = f"sample_size={sample_size}, threshold={threshold}, fraud_ratio={fraud_ratio}, scenario={stress_scenario}, models={models_used}"

    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO experiments (experiment_name, description, status)
            VALUES (?, ?, 'running')
            """,
            (experiment_name, description),
        )
        conn.commit()

        experiment_id = cursor.lastrowid

        # Сохраняем параметры в таблицу experiment_params
        params = {
            "sample_size": sample_size,
            "threshold": threshold,
            "fraud_ratio": fraud_ratio,
            "stress_scenario": stress_scenario,
            "models_used": str(models_used),
        }
        save_experiment_params(experiment_id, params)

        return experiment_id


def save_model_results(
    exp_id: int,
    model_name: str,
    mode: str,
    precision: float,
    recall: float,
    f1: float,
    business_cost: float,
    accuracy: float = None,
    roc_auc: float = None,
) -> None:
    """
    Сохраняет результаты модели в базу данных

    Parameters:
    -----------
    exp_id : int
        ID эксперимента
    model_name : str
        Название модели
    mode : str
        Режим ('classic' или 'stress')
    precision : float
        Точность
    recall : float
        Полнота
    f1 : float
        F1-мера
    business_cost : float
        Бизнес-стоимость
    accuracy : float, optional
        Точность (accuracy)
    roc_auc : float, optional
        ROC-AUC
    """
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO model_results (
                experiment_id, model_name, scenario,
                precision, recall, f1_score, business_cost, accuracy, roc_auc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                exp_id,
                model_name,
                mode,
                precision,
                recall,
                f1,
                business_cost,
                accuracy,
                roc_auc,
            ),
        )
        conn.commit()


def save_experiment_params(experiment_id: int, params: dict) -> None:
    """
    Сохраняет параметры эксперимента

    Parameters:
    -----------
    experiment_id : int
        ID эксперимента
    params : dict
        Словарь с параметрами (sample_size, fraud_ratio и т.д.)
    """
    with get_connection() as conn:
        cursor = conn.cursor()
        for param_name, param_value in params.items():
            if param_value is not None:
                cursor.execute(
                    """
                    INSERT INTO experiment_params (experiment_id, param_name, param_value)
                    VALUES (?, ?, ?)
                    """,
                    (experiment_id, param_name, str(param_value)),
                )
        conn.commit()


def finish_experiment(experiment_id: int, status: str = "completed") -> None:
    """
    Завершает эксперимент (меняет статус)

    Parameters:
    -----------
    experiment_id : int
        ID эксперимента
    status : str
        Статус ('completed', 'failed', 'cancelled')
    """
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE experiments 
            SET status = ? 
            WHERE id = ?
            """,
            (status, experiment_id),
        )
        conn.commit()


def get_experiment_results(experiment_id: int) -> pd.DataFrame:
    """
    Получает результаты эксперимента в виде DataFrame

    Parameters:
    -----------
    experiment_id : int
        ID эксперимента

    Returns:
    --------
    pd.DataFrame : Результаты всех моделей в эксперименте
    """
    query = """
        SELECT 
            mr.model_name,
            mr.scenario,
            mr.accuracy,
            mr.precision,
            mr.recall,
            mr.f1_score,
            mr.roc_auc,
            mr.business_cost,
            mr.created_at
        FROM model_results mr
        WHERE mr.experiment_id = ?
        ORDER BY mr.model_name, mr.scenario
    """

    with get_connection() as conn:
        return pd.read_sql_query(query, conn, params=(experiment_id,))


def get_all_experiments() -> pd.DataFrame:
    """
    Получает список всех экспериментов

    Returns:
    --------
    pd.DataFrame : Список экспериментов
    """
    query = """
        SELECT 
            id,
            experiment_name,
            experiment_date,
            description,
            status
        FROM experiments
        ORDER BY experiment_date DESC
    """

    with get_connection() as conn:
        return pd.read_sql_query(query, conn)


def get_experiment_summary(experiment_id: int) -> dict:
    """
    Получает сводку по эксперименту

    Parameters:
    -----------
    experiment_id : int
        ID эксперимента

    Returns:
    --------
    dict : Сводная информация
    """
    with get_connection() as conn:
        cursor = conn.cursor()

        # Информация об эксперименте
        cursor.execute("SELECT * FROM experiments WHERE id = ?", (experiment_id,))
        experiment = cursor.fetchone()

        # Количество моделей
        cursor.execute(
            """
            SELECT COUNT(DISTINCT model_name) as model_count 
            FROM model_results 
            WHERE experiment_id = ?
            """,
            (experiment_id,),
        )
        model_count = cursor.fetchone()["model_count"]

        # Количество записей
        cursor.execute(
            """
            SELECT COUNT(*) as total_results 
            FROM model_results 
            WHERE experiment_id = ?
            """,
            (experiment_id,),
        )
        total_results = cursor.fetchone()["total_results"]

        return {
            "experiment_id": experiment["id"],
            "experiment_name": experiment["experiment_name"],
            "experiment_date": experiment["experiment_date"],
            "status": experiment["status"],
            "model_count": model_count,
            "total_results": total_results,
        }


def delete_experiment(experiment_id: int) -> None:
    """
    Удаляет эксперимент и все связанные с ним данные

    Parameters:
    -----------
    experiment_id : int
        ID эксперимента
    """
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM experiments WHERE id = ?", (experiment_id,))
        conn.commit()


# Пример использования
if __name__ == "__main__":
    # Инициализация БД
    init_db()

    # Создание эксперимента
    exp_id = create_experiment(
        experiment_name="Тестовый запуск",
        description="Проверка работы БД",
        sample_size=1000,
        fraud_ratio=0.01,
        stress_scenario="normal",
    )
    print(f"Создан эксперимент с ID: {exp_id}")

    # Сохранение параметров
    save_experiment_params(
        exp_id, {"sample_size": 1000, "fraud_ratio": 0.01, "scenario": "normal"}
    )

    # Сохранение результатов модели
    save_model_results(
        exp_id=exp_id,
        model_name="Logistic Regression",
        mode="classic",
        precision=0.89,
        recall=0.92,
        f1=0.90,
        business_cost=1250,
        accuracy=0.95,
        roc_auc=0.97,
    )

    # Завершение эксперимента
    finish_experiment(exp_id)

    # Получение результатов
    results = get_experiment_results(exp_id)
    print("\nРезультаты:")
    print(results)

    # Список экспериментов
    experiments = get_all_experiments()
    print("\nВсе эксперименты:")
    print(experiments)

    # Сводка
    summary = get_experiment_summary(exp_id)
    print("\nСводка:")
    print(summary)
