"""
Модуль валидации CSV-файлов.

Содержит функции для проверки загруженных пользователем CSV-файлов:
- validate_csv() - основная функция валидации
- prepare_data_for_prediction() - подготовка данных для моделей
- get_column_info() - информация об ожидаемых колонках
"""

import numpy as np
import pandas as pd

from data_generator.generator import get_expected_columns


def validate_csv(df, require_target=False):
    """
    Проверяет валидность загруженного CSV файла.

    Параметры:
    - df: pandas DataFrame с данными от пользователя
    - require_target: bool, требуется ли колонка 'is_fraud'
                      True - для синтетических данных
                      False - для пользовательских CSV (по умолчанию)

    Возвращает:
    - is_valid: bool (True если всё корректно)
    - errors: список строк с описанием ошибок
    - warnings: список строк с предупреждениями
    """
    errors = []
    warnings = []

    # Получаем ожидаемые колонки из генератора
    expected_columns = get_expected_columns()

    # Если не требуем целевую переменную, убираем её из проверки
    if not require_target and "is_fraud" in expected_columns:
        expected_columns = [col for col in expected_columns if col != "is_fraud"]

    # 1. Проверка наличия всех обязательных колонок
    missing_cols = set(expected_columns) - set(df.columns)
    if missing_cols:
        errors.append(
            f"Отсутствуют обязательные столбцы: {', '.join(sorted(missing_cols))}"
        )

    # 2. Проверка на лишние колонки (предупреждение)
    extra_cols = set(df.columns) - set(expected_columns)
    # Если не требуется is_fraud, игнорируем её как лишнюю
    if not require_target and "is_fraud" in extra_cols:
        extra_cols.discard("is_fraud")
    if extra_cols:
        warnings.append(
            f"Обнаружены лишние столбцы: {', '.join(sorted(extra_cols))}. "
            "Они будут проигнорированы."
        )

    # 3. Проверка типов данных (должны быть числовые)
    # Список колонок, которые могут быть нечисловыми
    NON_NUMERIC_COLUMNS = ['category']

    # 3. Проверка типов данных (должны быть числовые, кроме исключений)
    for col in expected_columns:
        if col in df.columns:
            if col not in NON_NUMERIC_COLUMNS and not pd.api.types.is_numeric_dtype(df[col]):
                errors.append(
                    f"Столбец '{col}' должен быть числовым, "
                    f"но получен тип {df[col].dtype}"
                )

    # 4. Проверка на пропуски (null значения)
    for col in expected_columns:
        if col in df.columns and df[col].isnull().any():
            null_count = df[col].isnull().sum()
            errors.append(f"Столбец '{col}' содержит {null_count} пропущенных значений")

    # 5. Проверка на бесконечные значения (inf, -inf)
    for col in expected_columns:
        if col in df.columns and df[col].dtype in ["float64", "float32"]:
            if np.isinf(df[col]).any():
                inf_count = np.isinf(df[col]).sum()
                errors.append(
                    f"Столбец '{col}' содержит {inf_count} бесконечных значений"
                )

    # 6. Проверка, что в данных есть хотя бы одна строка
    if len(df) == 0:
        errors.append("Файл не содержит данных (0 строк)")

    # 7. Проверка, что нет дубликатов строк (опционально)
    if df.duplicated().any():
        dup_count = df.duplicated().sum()
        warnings.append(f"Обнаружено {dup_count} дублирующихся строк")

    is_valid = len(errors) == 0

    return is_valid, errors, warnings


def prepare_data_for_prediction(df):
    """
    Подготавливает DataFrame для предсказания.

    - Оставляет только нужные колонки
    - Заполняет пропуски (если есть) - лучше чтобы валидатор их отловил

    Параметры:
    - df: исходный DataFrame

    Возвращает:
    - X: DataFrame только с признаками для модели
    """
    expected_columns = get_expected_columns()

    # Оставляем только ожидаемые колонки
    available_cols = [col for col in expected_columns if col in df.columns]
    X = df[available_cols].copy()

    # Если есть пропуски - заполняем медианой (но лучше чтобы валидатор их отловил)
    for col in X.columns:
        if X[col].isnull().any():
            X[col].fillna(X[col].median(), inplace=True)

    return X


def get_column_info():
    """Возвращает информацию об ожидаемых колонках для отображения в UI."""
    expected_columns = get_expected_columns()

    return {
        "expected_columns": expected_columns,
        "total_count": len(expected_columns),
        "description": "Признаки для модели обнаружения фрода",
        "sample_types": {
            "amount": "числовой (сумма транзакции)",
            "transaction_minute": "числовой (минуты с 00:00)",
            "day_of_week": "числовой (0-6)",
            "distance_km": "числовой",
            "is_contactless": "бинарный (0/1)",
            "age": "числовой (возраст клиента)",
            "device_risk": "категориальный (0, 1, 2)",
        },
    }
