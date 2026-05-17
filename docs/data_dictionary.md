## ER-диаграмма сущности транзакции

```mermaid
erDiagram
    CLIENT {
        int client_id PK
        int age
        int city_population
        float home_lat
        float home_lon
        float avg_amount
        float std_amount
        int typical_minute
        bool phone_changed
        float device_risk
    }
    
    TRANSACTION {
        int transaction_id PK
        int client_id FK
        datetime timestamp
        string category
        float amount
        int minute_of_day
        int day_of_week
        int time_deviation
        bool unusual_time_flag
        float distance_from_home
        float nfc_duration_ms
        int device_age_days
        bool is_new_device
        bool amount_deviation_flag
        bool unusual_sms_flag
        bool phone_recently_changed
        bool suspicious_deposit
        bool new_beneficiary
        bool is_fraud
        bool observed_fraud
    }
    
    CLIENT ||--o{ TRANSACTION : has

# Словарь данных транзакции

## Статичные признаки клиента (на уровне клиента)

| Признак | Тип | Описание | Диапазон / Значения |
|---------|-----|----------|---------------------|
| `client_age` | int | Возраст клиента | 18–80 лет |
| `city_population` | int | Численность населения города проживания (условная) | 1000 – 12 млн |
| `home_lat` | float | Географическая широта домашнего местоположения | -90 – 90 |
| `home_lon` | float | Географическая долгота домашнего местоположения | -180 – 180 |
| `avg_transaction_amount` | float | Средняя сумма транзакции клиента (лог-нормальное распределение) | >0 |
| `std_transaction_amount` | float | Стандартное отклонение суммы транзакций клиента | >0 |
| `typical_transaction_minute` | int | Типичное время совершения транзакции (минуты от 00:00) | 0–1439 |
| `phone_number_changed` | bool | Была ли смена телефонного номера (вероятность 5%) | 0 или 1 |
| `device_risk_score` | float | Искусственный показатель риска устройства (новизна, история) | 0–1 |

## Динамические признаки транзакции

| Признак | Тип | Описание | Диапазон / Значения |
|---------|-----|----------|---------------------|
| `transaction_id` | int | Уникальный идентификатор транзакции | >0 |
| `client_id` | int | Идентификатор клиента | 1–5000 |
| `timestamp` | datetime | Время совершения транзакции | 2025-01-01 – 2026-06-30 |
| `transaction_category` | categorical | Категория транзакции | продукты, развлечения, путешествия, онлайн-шоппинг, рестораны, транспорт, наличные, другое |
| `transaction_amount` | float | Сумма транзакции | >0 |
| `minute_of_day` | int | Минута совершения транзакции (извлечено из timestamp) | 0–1439 |
| `day_of_week` | int | День недели (0=понедельник) | 0–6 |
| `time_deviation` | int | Циклическое отклонение от типичного времени клиента (минуты) | 0–720 |
| `unusual_time_flag` | bool | Признак неординарного времени (отклонение >420 минут) | 0 или 1 |
| `distance_from_home` | float | Расстояние между домашним адресом и местом транзакции (км) | ≥0 |
| `nfc_duration_ms` | float | Длительность NFC-сессии (мс) | 300–800 |
| `device_age_days` | int | Возраст устройства (дней) | 0–730 |
| `is_new_device` | bool | Признак нового устройства (<90 дней) | 0 или 1 |

## Сигнальные признаки (слабые сигналы мошенничества)

| Признак | Тип | Описание | Диапазон / Значения |
|---------|-----|----------|---------------------|
| `amount_deviation_flag` | bool | Отклонение суммы >2.5 std от среднего по клиенту | 0 или 1 |
| `unusual_sms_flag` | bool | Необычные SMS-оповещения за последние 6 часов | 0 или 1 |
| `phone_recently_changed_flag` | bool | Смена номера телефона за последние 48 часов | 0 или 1 |
| `suspicious_deposit_flag` | bool | Подозрительный депозит наличных (при NFC) | 0 или 1 |
| `new_beneficiary_self_transfer` | bool | Новый бенефициар после перевода самому себе | 0 или 1 |

## Целевая переменная

| Признак | Тип | Описание |
|---------|-----|----------|
| `is_fraud` | bool (int) | Является ли транзакция мошеннической (истинная метка) |
| `observed_fraud` | bool (int) | Наблюдаемая метка (с шумом, 2.5% фрода) |
