"""Общие утилиты для обучения моделей."""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import warnings

warnings.filterwarnings('ignore')


def load_data(data_path: str, exclude_cols: list = None) -> tuple:
    """Загружает и подготавливает данные."""
    if exclude_cols is None:
        exclude_cols = ['client_id', 'trans_date', 'merch_lat', 'merch_lon', 'is_fraud']
    
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"   Shape: {df.shape}")
    print(f"   Fraud ratio: {df['is_fraud'].mean():.4f}")
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    X = df[feature_cols].copy()
    y = df['is_fraud'].values
    
    return X, y, df, feature_cols


def split_data(X, y, test_size=0.2, random_state=42, stratify=True):
    """Разделяет данные на train/test."""
    stratify_param = y if stratify else None
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify_param
    )


def get_categorical_features(X):
    """Определяет категориальные признаки."""
    categorical_features = []
    for col in X.columns:
        if X[col].dtype == 'object' or X[col].dtype.name == 'category':
            categorical_features.append(col)
    return categorical_features


def train_logistic_regression(X_train, y_train, **kwargs):
    """Обучает логистическую регрессию."""
    default_params = {
        'random_state': 42,
        'max_iter': 1000,
        'class_weight': 'balanced'
    }
    default_params.update(kwargs)
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(**default_params))
    ])
    pipeline.fit(X_train, y_train)
    return pipeline


def train_random_forest(X_train, y_train, **kwargs):
    """Обучает случайный лес."""
    default_params = {
        'n_estimators': 100,
        'random_state': 42,
        'class_weight': 'balanced',
        'n_jobs': -1
    }
    default_params.update(kwargs)
    
    model = RandomForestClassifier(**default_params)
    model.fit(X_train, y_train)
    return model


def train_catboost(X_train, y_train, categorical_features=None, **kwargs):
    """Обучает CatBoost."""
    default_params = {
        'iterations': 200,
        'depth': 6,
        'learning_rate': 0.1,
        'random_seed': 42,
        'verbose': False,
        'class_weights': [1, 10]
    }
    default_params.update(kwargs)
    
    if categorical_features:
        default_params['cat_features'] = categorical_features
    
    # Конвертируем категориальные признаки в строки
    X_train = X_train.copy()
    for col in categorical_features or []:
        if col in X_train.columns:
            X_train[col] = X_train[col].astype(str)
    
    model = CatBoostClassifier(**default_params)
    model.fit(X_train, y_train)
    return model


def train_isolation_forest(X_train_normal, **kwargs):
    """Обучает Isolation Forest на нормальных транзакциях."""
    default_params = {
        'n_estimators': 100,
        'contamination': 0.025,
        'random_state': 42
    }
    default_params.update(kwargs)
    
    model = IsolationForest(**default_params)
    model.fit(X_train_normal)
    return model


def prepare_for_isolation_forest(X, categorical_features=None):
    """Подготавливает данные для Isolation Forest (one-hot encoding)."""
    X_processed = X.copy()
    if categorical_features:
        X_processed = pd.get_dummies(X_processed, columns=categorical_features, drop_first=True)
    else:
        X_processed = pd.get_dummies(X_processed)
    return X_processed


def evaluate_model(model, X_test, y_test, model_name=""):
    """Оценивает модель и возвращает метрики."""
    if hasattr(model, "predict_proba"):
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_proba)
    else:
        y_pred = model.predict(X_test)
        roc_auc = None
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc
    }
    
    if model_name:
        print(f"\n{model_name}:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1:        {metrics['f1']:.4f}")
        if roc_auc:
            print(f"  ROC-AUC:   {roc_auc:.4f}")
    
    return metrics


def save_model(model, path: str):
    """Сохраняет модель в файл."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"   Model saved: {path}")
