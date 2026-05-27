# =====================================================
# СТАНДАРТНЫЕ БИБЛИОТЕКИ
# =====================================================
from datetime import datetime, timedelta
import random

# =====================================================
# СТОРОННИЕ БИБЛИОТЕКИ
# =====================================================
import numpy as np
import pandas as pd
from faker import Faker


# =====================================================
# ГЕНЕРАТОР ДАННЫХ
# =====================================================

def generate_fraud_dataset(
    n_transactions: int = 2000,
    fraud_ratio: float = 0.01,
    label_noise: float = 0.015,
    random_state: int = 42
) -> tuple:
    """
    Генерирует синтетический датасет транзакций.

    Parameters
    ----------
    n_transactions : int
        Количество транзакций (по умолчанию 2000)
    fraud_ratio : float
        Доля мошеннических транзакций до шума (по умолчанию 0.01 = 1%)
    label_noise : float
        Доля перевёрнутых меток (по умолчанию 0.015 = 1.5%)
    random_state : int
        Seed для воспроизводимости (по умолчанию 42)

    Returns
    -------
    tuple
        (X, y, df) где:
        - X : pd.DataFrame — признаки
        - y : pd.Series — метки
        - df : pd.DataFrame — полный датасет с признаками и меткой
    """
    np.random.seed(random_state)
    random.seed(random_state)
    fake = Faker('ru_RU')

    start_date = datetime(2025, 1, 1)
    end_date = datetime(2026, 6, 30)
    delta_days = (end_date - start_date).days
    SECONDS_PER_DAY = 86400

    # Генерация базы клиентов
    n_clients = 5000
    clients_data = []
    for client_id in range(1, n_clients + 1):
        age = np.random.randint(18, 81)
        lat, lng = map(float, fake.latlng())
        city_pop = np.random.choice([5000, 50000, 500000, 5000000], p=[0.2, 0.3, 0.3, 0.2])
        avg_amount = np.random.lognormal(mean=7.0, sigma=0.8)
        cv = np.random.lognormal(mean=-0.7, sigma=0.5)
        std_amount = avg_amount * cv

        typical_hour = np.clip(np.random.normal(14, 3), 0, 23)
        typical_minute = int(np.random.randint(0, 60))
        typical_minute_client = typical_hour * 60 + typical_minute

        device_risk = np.random.choice([0, 1, 2], p=[0.85, 0.10, 0.05])
        phone_last_change = None
        if np.random.rand() < 0.05:
            phone_last_change = start_date + timedelta(days=np.random.randint(0, delta_days))

        clients_data.append({
            'client_id': client_id,
            'age': age,
            'city_pop': city_pop,
            'home_lat': lat,
            'home_lon': lng,
            'avg_amount_client': avg_amount,
            'std_amount_client': std_amount,
            'typical_minute_client': typical_minute_client,
            'device_risk': device_risk,
            'phone_last_change': phone_last_change,
        })
    clients = pd.DataFrame(clients_data)
    clients['home_lat'] = clients['home_lat'].astype(float)
    clients['home_lon'] = clients['home_lon'].astype(float)

    # Генерация транзакций
    n_legit = int(n_transactions * (1 - fraud_ratio))
    n_fraud = n_transactions - n_legit

    transactions = []

    for _ in range(n_legit):
        client_id = np.random.choice(clients['client_id'])
        ts = start_date + timedelta(seconds=np.random.randint(0, delta_days * SECONDS_PER_DAY))
        transactions.append({'client_id': client_id, 'trans_date': ts, 'is_fraud': 0})

    for _ in range(n_fraud):
        client_id = np.random.choice(clients['client_id'])
        ts = start_date + timedelta(seconds=np.random.randint(0, delta_days * SECONDS_PER_DAY))
        transactions.append({'client_id': client_id, 'trans_date': ts, 'is_fraud': 1})

    df = pd.DataFrame(transactions)
    df = df.sort_values(['client_id', 'trans_date']).reset_index(drop=True)
    df = df.merge(clients, on='client_id', how='left')

    df['transaction_minute'] = df['trans_date'].dt.hour * 60 + df['trans_date'].dt.minute
    df['day_of_week'] = df['trans_date'].dt.dayofweek

    def cyclic_deviation(h1, h2):
        diff = np.abs(h1 - h2)
        return np.minimum(diff, 1440 - diff)
    df['time_deviation_min'] = cyclic_deviation(df['transaction_minute'].values, df['typical_minute_client'].values)
    df['is_unusual_time'] = (df['time_deviation_min'] > 420).astype(int)

    categories = ['grocery', 'entertainment', 'travel', 'online_shopping', 'restaurant', 'transport', 'other', 'cash_deposit']
    def get_category(is_fraud):
        if is_fraud == 0:
            return np.random.choice(categories, p=[0.2,0.15,0.1,0.2,0.15,0.1,0.09,0.01])
        else:
            return np.random.choice(categories, p=[0.15,0.1,0.2,0.3,0.1,0.05,0.05,0.05])
    df['category'] = df['is_fraud'].apply(get_category)

    legit_mask = df['is_fraud'] == 0
    fraud_mask = df['is_fraud'] == 1

    shape = df.loc[legit_mask, 'avg_amount_client']**2 / (df.loc[legit_mask, 'std_amount_client']**2 + 1e-6)
    scale = (df.loc[legit_mask, 'std_amount_client']**2 + 1e-6) / (df.loc[legit_mask, 'avg_amount_client'] + 1e-6)
    df.loc[legit_mask, 'amount'] = np.random.gamma(shape, scale)

    fraud_avg = df.loc[fraud_mask, 'avg_amount_client'].values
    factors = np.where(np.random.rand(len(fraud_avg)) < 0.5,
                       np.random.uniform(3, 7, len(fraud_avg)),
                       np.random.uniform(0.3, 0.8, len(fraud_avg)))
    df.loc[fraud_mask, 'amount'] = fraud_avg * factors

    df['is_contactless'] = 0
    high_cats = ['grocery', 'restaurant', 'transport']
    for cat in high_cats:
        mask = (df['category'] == cat) & (df['is_fraud'] == 0)
        df.loc[mask, 'is_contactless'] = np.random.binomial(1, 0.7, size=mask.sum())
    mask = (df['category'].isin(high_cats)) & (df['is_fraud'] == 1)
    df.loc[mask, 'is_contactless'] = np.random.binomial(1, 0.8, size=mask.sum())
    mask_low = (~df['category'].isin(high_cats)) & (df['is_fraud'] == 0)
    df.loc[mask_low, 'is_contactless'] = np.random.binomial(1, 0.25, size=mask_low.sum())
    mask_low_fraud = (~df['category'].isin(high_cats)) & (df['is_fraud'] == 1)
    df.loc[mask_low_fraud, 'is_contactless'] = np.random.binomial(1, 0.35, size=mask_low_fraud.sum())

    contactless = df['is_contactless'] == 1
    df['nfc_duration_ms'] = 0.0
    df.loc[contactless & (df['is_fraud']==0), 'nfc_duration_ms'] = np.random.normal(450, 150, size=(contactless & (df['is_fraud']==0)).sum())
    df.loc[contactless & (df['is_fraud']==1), 'nfc_duration_ms'] = np.random.normal(550, 200, size=(contactless & (df['is_fraud']==1)).sum())
    df['nfc_time_exceeded'] = (df['nfc_duration_ms'] > 500).astype(int)

    dist_km = np.zeros(len(df))
    for cat in ['travel', 'online_shopping']:
        mask = (df['category'] == cat) & (df['is_fraud']==0)
        dist_km[mask] = np.random.exponential(scale=80, size=mask.sum())
    mask_other = (df['is_fraud']==0) & (~df['category'].isin(['travel','online_shopping']))
    dist_km[mask_other] = np.random.exponential(scale=15, size=mask_other.sum())
    mask_fraud = (df['is_fraud']==1)
    dist_km[mask_fraud] = np.random.exponential(scale=100, size=mask_fraud.sum()) + 10

    home_lat = df['home_lat'].values.astype(float)
    home_lon = df['home_lon'].values.astype(float)
    angle = np.random.uniform(0, 2*np.pi, size=len(df))
    lat_offset = dist_km / 111.0
    lon_offset = dist_km / (111.0 * np.cos(np.radians(home_lat)))
    df['merch_lat'] = home_lat + lat_offset * np.cos(angle)
    df['merch_lon'] = home_lon + lon_offset * np.sin(angle)
    df['distance_km'] = dist_km

    avg = df['avg_amount_client']
    std = df['std_amount_client']
    amount = df['amount']
    df['is_unusual_amount'] = ((amount > avg + 2.5*std) | (amount < avg - 2.5*std)).astype(int)

    base_sms_prob = np.where(df['is_fraud'] == 1, 0.25, 0.06)
    client_sms_noise = np.random.normal(0, 0.08, len(df))
    sms_prob = np.clip(base_sms_prob + client_sms_noise, 0.01, 0.45)
    df['sms_anomaly_6h'] = np.random.binomial(1, sms_prob)

    def phone_changed(row):
        if pd.isna(row['phone_last_change']):
            return 0
        diff_hours = (row['trans_date'] - row['phone_last_change']).total_seconds() / 3600
        return 1 if 0 <= diff_hours <= 48 else 0
    df['phone_changed_48h'] = df.apply(phone_changed, axis=1)

    cash_mask = (df['category'] == 'cash_deposit') & (df['is_contactless'] == 1)
    base_prob_cash = np.where(df['is_fraud'] == 1, 0.3, 0.04)
    noise_cash = np.random.normal(0, 0.1, cash_mask.sum())
    prob_cash = np.clip(base_prob_cash[cash_mask] + noise_cash, 0.01, 0.5)
    df['suspect_cash_deposit'] = 0
    df.loc[cash_mask, 'suspect_cash_deposit'] = np.random.binomial(1, prob_cash)

    base_self_prob = np.where(df['is_fraud'] == 1, 0.25, 0.04)
    self_noise = np.random.normal(0, 0.08, len(df))
    self_prob = np.clip(base_self_prob + self_noise, 0.01, 0.4)
    df['new_beneficiary_after_self_transfer'] = np.random.binomial(1, self_prob)

    df['device_age_days'] = np.random.exponential(365, len(df)) * (1 + 0.3*df['is_fraud'] + np.random.normal(0, 0.2, len(df)))
    df['device_risk_high'] = (df['device_age_days'] < 90).astype(int)

    flip_mask = np.random.rand(len(df)) < label_noise
    df.loc[flip_mask, 'is_fraud'] = 1 - df.loc[flip_mask, 'is_fraud']

    feature_cols = ['amount', 'transaction_minute', 'day_of_week', 'time_deviation_min',
                    'distance_km', 'nfc_time_exceeded', 'nfc_duration_ms', 'is_unusual_amount',
                    'is_unusual_time', 'sms_anomaly_6h', 'phone_changed_48h', 'device_risk_high',
                    'device_age_days', 'suspect_cash_deposit', 'new_beneficiary_after_self_transfer',
                    'is_contactless', 'age', 'city_pop', 'avg_amount_client', 'std_amount_client',
                    'typical_minute_client', 'device_risk']

    X = df[feature_cols].copy()
    y = df['is_fraud'].copy()

    return X, y, df[feature_cols + ['is_fraud']]


# =====================================================
# ФУНКЦИИ ДЛЯ ПОЛУЧЕНИЯ СПИСКОВ КОЛОНОК
# =====================================================

def get_expected_columns() -> list:
    """Возвращает список колонок-признаков, которые генерирует generate_fraud_dataset."""
    return [
        'amount',
        'transaction_minute',
        'day_of_week',
        'time_deviation_min',
        'distance_km',
        'nfc_time_exceeded',
        'nfc_duration_ms',
        'is_unusual_amount',
        'is_unusual_time',
        'sms_anomaly_6h',
        'phone_changed_48h',
        'device_risk_high',
        'device_age_days',
        'suspect_cash_deposit',
        'new_beneficiary_after_self_transfer',
        'is_contactless',
        'age',
        'city_pop',
        'avg_amount_client',
        'std_amount_client',
        'typical_minute_client',
        'device_risk'
    ]


def get_numeric_features() -> list:
    """
    Возвращает список числовых признаков (без 'category')
    для моделей Logistic Regression и Random Forest.
    """
    all_features = get_expected_columns()
    return [col for col in all_features if col != 'category']


def get_features_for_iforest() -> list:
    """
    Возвращает список признаков для Isolation Forest
    (с учётом one-hot encoding категории 'category').
    """
    numeric_features = get_numeric_features()
    
    categories = [
        'category_grocery', 'category_entertainment', 'category_travel',
        'category_online_shopping', 'category_restaurant', 'category_transport',
        'category_other', 'category_cash_deposit'
    ]
    
    return numeric_features + categories


# =====================================================
# ФУНКЦИЯ ПОДВЫБОРКИ
# =====================================================

def generate_fraud_subset(
    subset_size: int = 500,
    full_size: int = 2000,
    fraud_ratio: float = 0.01,
    label_noise: float = 0.015,
    random_state: int = 42,
    use_stratification: bool = True
) -> pd.DataFrame:
    """
    Генерирует подвыборку указанного размера (100-2000) из основного генератора.

    Parameters
    ----------
    subset_size : int
        Размер подвыборки (от 100 до 2000, по умолчанию 500)
    full_size : int
        Размер полного датасета (по умолчанию 2000)
    fraud_ratio : float
        Доля мошеннических транзакций (по умолчанию 0.01 = 1%)
    label_noise : float
        Доля перевёрнутых меток (по умолчанию 0.015 = 1.5%)
    random_state : int
        Seed для воспроизводимости (по умолчанию 42)
    use_stratification : bool
        Сохранять точную долю фрода (по умолчанию True)

    Returns
    -------
    pd.DataFrame
        Подвыборка с признаками и колонкой 'is_fraud'

    Raises
    ------
    ValueError
        Если subset_size не в диапазоне 100-2000 или больше full_size
    """
    if not 100 <= subset_size <= 2000:
        raise ValueError(f"subset_size должен быть от 100 до 2000, получено {subset_size}")
    
    if subset_size > full_size:
        raise ValueError(f"subset_size ({subset_size}) не может быть больше full_size ({full_size})")
    
    _, _, full_df = generate_fraud_dataset(
        n_transactions=full_size,
        fraud_ratio=fraud_ratio,
        label_noise=label_noise,
        random_state=random_state
    )
    
    if use_stratification:
        n_fraud = int(subset_size * fraud_ratio)
        n_legit = subset_size - n_fraud
        
        fraud_df = full_df[full_df['is_fraud'] == 1]
        legit_df = full_df[full_df['is_fraud'] == 0]
        
        actual_n_fraud = min(n_fraud, len(fraud_df))
        actual_n_legit = min(n_legit, len(legit_df))
        
        if len(fraud_df) < n_fraud:
            actual_n_legit += (n_fraud - len(fraud_df))
        if len(legit_df) < n_legit:
            actual_n_fraud += (n_legit - len(legit_df))
        
        fraud_sample = fraud_df.sample(n=actual_n_fraud, random_state=random_state)
        legit_sample = legit_df.sample(n=actual_n_legit, random_state=random_state)
        
        subset_df = pd.concat([fraud_sample, legit_sample])
        subset_df = subset_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    else:
        subset_df = full_df.sample(n=subset_size, random_state=random_state).reset_index(drop=True)
    
    return subset_df