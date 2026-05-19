import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix


def compute_metrics(y_true, y_pred, y_proba=None, cost_fp=1, cost_fn=10):
    """Расчёт метрик: Accuracy, Precision, Recall, F1, ROC-AUC, Business Cost"""
    metrics = {}
    
    metrics['Accuracy'] = accuracy_score(y_true, y_pred)
    metrics['Precision'] = precision_score(y_true, y_pred, zero_division=0)
    metrics['Recall'] = recall_score(y_true, y_pred, zero_division=0)
    metrics['F1'] = f1_score(y_true, y_pred, zero_division=0)
    
    if y_proba is not None:
        metrics['ROC_AUC'] = roc_auc_score(y_true, y_proba)
    
    # Матрица ошибок: TN, FP, FN, TP
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    # Бизнес-стоимость: FN дороже FP
    total_cost = fp * cost_fp + fn * cost_fn
    metrics['Business_Cost'] = total_cost
    
    metrics['FP_count'] = fp
    metrics['FN_count'] = fn
    metrics['TP_count'] = tp
    metrics['TN_count'] = tn
    metrics['Avg_Cost_per_Transaction'] = total_cost / len(y_true)
    
    return metrics


def format_metrics_table(metrics_dict):
    """Форматирует словарь с метриками в DataFrame"""
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


def evaluate_with_stress(model, X, y_true, scenario_names, cost_fp=1, cost_fn=10):
    """Оценивает модель на стресс-сценариях (imbalance, amount_shift, masking, frequency_boost)"""
    from data_generator.stress_scenarios import apply_stress
    
    results = {}
    df_full = X.copy()
    df_full['is_fraud'] = y_true.values if hasattr(y_true, 'values') else y_true
    
    for scenario in scenario_names:
        # Применяем стресс к данным
        df_stressed = apply_stress(df_full, scenario)
        X_stressed = df_stressed.drop('is_fraud', axis=1)
        y_stressed = df_stressed['is_fraud']
        
        # Предсказания модели
        y_pred = model.predict(X_stressed)
        y_proba = model.predict_proba(X_stressed)[:, 1] if hasattr(model, "predict_proba") else None
        
        results[scenario] = compute_metrics(y_stressed, y_pred, y_proba, cost_fp, cost_fn)
    
    return results


def evaluate_all_models(models_dict, X, y_true, scenario_names, cost_fp=1, cost_fn=10):
    """Оценивает все модели на всех стресс-сценариях"""
    all_results = {}
    for model_name, model in models_dict.items():
        all_results[model_name] = evaluate_with_stress(model, X, y_true, scenario_names, cost_fp, cost_fn)
    return all_results


def calculate_drawdown(normal_metrics, stress_metrics):
    """Рассчитывает просадку метрик в процентах: (стресс - норма) / норма * 100%"""
    drawdown = {}
    for metric in normal_metrics:
        if metric in stress_metrics and normal_metrics[metric] != 0:
            # Пропускаем счётчики (FP_count и т.д.)
            if not metric.endswith('_count') and metric != 'Business_Cost':
                drawdown[metric] = round(((stress_metrics[metric] - normal_metrics[metric]) / normal_metrics[metric]) * 100, 2)
            elif metric == 'Business_Cost':
                drawdown[metric] = round(((stress_metrics[metric] - normal_metrics[metric]) / normal_metrics[metric]) * 100, 2)
    return drawdown


def format_results_with_drawdown(results):
    """Форматирует результаты с просадкой в таблицу для отображения в UI"""
    rows = []
    for model_name, scenarios in results.items():
        normal_metrics = scenarios.get('normal', {})
        for scenario, stress_metrics in scenarios.items():
            if scenario == 'normal':
                continue
            drawdown = calculate_drawdown(normal_metrics, stress_metrics)
            row = {
                'Model': model_name,
                'Scenario': scenario,
                'Accuracy (normal)': round(normal_metrics.get('Accuracy', 0), 4),
                'Accuracy (stress)': round(stress_metrics.get('Accuracy', 0), 4),
                'Accuracy drawdown %': drawdown.get('Accuracy'),
                'F1 (normal)': round(normal_metrics.get('F1', 0), 4),
                'F1 (stress)': round(stress_metrics.get('F1', 0), 4),
                'F1 drawdown %': drawdown.get('F1'),
                'Cost (normal)': int(normal_metrics.get('Business_Cost', 0)),
                'Cost (stress)': int(stress_metrics.get('Business_Cost', 0)),
                'Cost drawdown %': drawdown.get('Business_Cost')
            }
            rows.append(row)
    return pd.DataFrame(rows)
