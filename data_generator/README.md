# Генератор синтетических данных

## Описание
Модуль `data_generator` предоставляет функции для генерации реалистичных синтетических данных транзакций.

## Автор
Сахарова (Backend 1)

---

## Функции

### `generate_fraud_dataset(n_transactions, fraud_ratio, label_noise, random_state)`

**Продвинутый генератор** с реалистичными признаками (Faker, геолокация, времена, категории).

**Параметры:**
| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| n_transactions | int | 2000 | Количество транзакций |
| fraud_ratio | float | 0.01 | Доля мошеннических транзакций (1%) |
| label_noise | float | 0.015 | Доля перевёрнутых меток (1.5%) |
| random_state | int | 42 | Seed для воспроизводимости |

**Возвращает:** `(X, y, df)`

**Пример:**
```python
from data_generator import generate_fraud_dataset
X, y, df = generate_fraud_dataset(n_transactions=10000, fraud_ratio=0.01)
```

### `generate_fraud_subset(subset_size, full_size, fraud_ratio, label_noise, random_state, use_stratification)`

**Генератор подвыборки** заданного размера (100–2000 строк).

**Параметры:**
| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| subset_size | int | 500 | Размер подвыборки (100-2000) |
| full_size | int | 2000 | Размер полного датасета |
| fraud_ratio | float | 0.01 | Доля мошеннических транзакций |
| use_stratification | bool | True | Сохранять точную долю фрода |

**Пример:**
```python
from data_generator import generate_fraud_subset
df = generate_fraud_subset(subset_size=500)
```

### `apply_stress(df, scenario)`

**Применяет стресс-сценарий** к данным.

**Сценарии:** `normal`, `imbalance`, `amount_shift`, `masking`, `frequency_boost`

**Пример:**
```python
from data_generator import generate_fraud_dataset, apply_stress
_, _, df = generate_fraud_dataset()
df_stressed = apply_stress(df, 'imbalance')
```

### `get_expected_columns()`

**Возвращает список ожидаемых колонок** для валидации CSV.

**Пример:**
```python
from data_generator import get_expected_columns
columns = get_expected_columns()
```
