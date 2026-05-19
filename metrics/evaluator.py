import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

def compute_metrics(y_true, y_pred, y_proba=None, cost_fp=1, cost_fn=10):
    """Расчёт всех метрик для моделей обнаружения фрода"""
    metrics = {}
    
    metrics['Accuracy'] = accuracy_score(y_true, y_pred)
    metrics['Precision'] = precision_score(y_true, y_pred, zero_division=0)
    metrics['Recall'] = recall_score(y_true, y_pred, zero_division=0)
    metrics['F1'] = f1_score(y_true, y_pred, zero_division=0)
    
    if y_proba is not None:
        metrics['ROC_AUC'] = roc_auc_score(y_true, y_proba)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    total_cost = fp * cost_fp + fn * cost_fn
    metrics['Business_Cost'] = total_cost
    
    metrics['FP_count'] = fp
    metrics['FN_count'] = fn
    metrics['TP_count'] = tp
    metrics['TN_count'] = tn
    metrics['Avg_Cost_per_Transaction'] = total_cost / len(y_true)
    
    return metrics

def format_metrics_table(metrics_dict):
    """Форматирует словарь с метриками для красивого вывода"""
    rows = []
    for model_name, metrics in metrics_dict.items():
        row = {'Model': model_name}
        for metric_name, value in metrics.items():
            if metric_name in ['Business_Cost', 'FP_count', 'FN_count', 'TP_count', 'TN_count']:
                row[metric_name] = int(value)
            elif metric_name == 'Avg_Cost_per_Transaction':
                row[metric_name] = round(value, 2)
            else:
                row[metric_name] = round(value, 4)
        rows.append(row)
    return pd.DataFrame(rows)