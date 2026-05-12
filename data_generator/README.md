# Генератор синтетических данных

## Описание
Модуль `data_generator` предоставляет функции для генерации синтетических данных транзакций.

## Автор
Сахарова (Backend 1)

## Функции

### `generate_transactions(n_samples, fraud_prob, random_state)`
**Простой генератор** с 10 числовыми признаками.

**Параметры:**
- `n_samples` (int, default=1000): Количество транзакций
- `fraud_prob` (float, default=0.05): Доля мошеннических транзакций
- `random_state` (int, default=42): Seed для воспроизводимости

**Пример:**
```python
from data_generator import generate_transactions
df = generate_transactions(n_samples=5000, fraud_prob=0.05)
```

### `generate_fraud_dataset(n_transactions, fraud_ratio, label_noise, random_state)`
**Продвинутый генератор** с реалистичными признаками.

**Параметры:**
- `n_transactions` (int, default=2000): Количество транзакций
- `fraud_ratio` (float, default=0.01): Доля мошеннических транзакций
- `label_noise` (float, default=0.015): Доля перевёрнутых меток
- `random_state` (int, default=42): Seed для воспроизводимости

**Пример:**
```python
from data_generator import generate_fraud_dataset
X, y, df = generate_fraud_dataset(n_transactions=10000, fraud_ratio=0.01)
```

### `generate_fraud_subset(subset_size, full_size, fraud_ratio, label_noise, random_state, use_stratification)`
**Генератор подвыборки** заданного размера (100–2000 строк).

**Параметры:**
- `subset_size` (int, default=500): Размер подвыборки (100-2000)
- `full_size` (int, default=2000): Размер полного датасета
- `fraud_ratio` (float, default=0.01): Доля мошеннических транзакций
- `random_state` (int, default=42): Seed для воспроизводимости
- `use_stratification` (bool, default=True): Сохранять точную долю фрода

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
