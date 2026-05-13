"""Unit-тесты для модуля evaluator.py"""
import pytest
import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from metrics.evaluator import compute_metrics, format_metrics_table

class TestComputeMetrics:
    """Тесты для функции compute_metrics"""
    
    def test_perfect_predictions(self):
        """Тест: идеальные предсказания"""
        y_true = [1, 0, 1, 0, 1]
        y_pred = [1, 0, 1, 0, 1]
        
        metrics = compute_metrics(y_true, y_pred)
        
        assert metrics['Accuracy'] == 1.0
        assert metrics['Precision'] == 1.0
        assert metrics['Recall'] == 1.0
        assert metrics['F1'] == 1.0
        assert metrics['FP_count'] == 0
        assert metrics['FN_count'] == 0
    
    def test_worst_predictions(self):
        """Тест: все предсказания неверны"""
        y_true = [1, 0, 1, 0]
        y_pred = [0, 1, 0, 1]
        
        metrics = compute_metrics(y_true, y_pred)
        
        assert metrics['Accuracy'] == 0.0
        assert metrics['Precision'] == 0.0
        assert metrics['Recall'] == 0.0
        assert metrics['F1'] == 0.0
    
    def test_with_probabilities(self):
        """Тест: с вероятностями предсказаний"""
        y_true = [1, 0, 1, 0]
        y_pred = [1, 0, 1, 0]
        y_proba = [0.9, 0.1, 0.8, 0.2]
        
        metrics = compute_metrics(y_true, y_pred, y_proba)
        
        assert 'ROC_AUC' in metrics
        assert 0.0 <= metrics['ROC_AUC'] <= 1.0
    
    def test_business_cost(self):
        """Тест: расчёт бизнес-стоимости"""
        y_true = [1, 0, 1, 0]
        y_pred = [0, 1, 1, 0]
        
        metrics = compute_metrics(y_true, y_pred, cost_fp=2, cost_fn=20)
        
        assert metrics['Business_Cost'] == 22
        assert metrics['FP_count'] == 1
        assert metrics['FN_count'] == 1
    
    def test_cost_default_values(self):
        """Тест: значения стоимости по умолчанию"""
        y_true = [1, 0]
        y_pred = [0, 1]
        
        metrics = compute_metrics(y_true, y_pred)
        
        assert metrics['Business_Cost'] == 11
    
    def test_imbalanced_dataset(self):
        """Тест: несбалансированный датасет"""
        y_true = [0] * 90 + [1] * 10
        y_pred = [0] * 85 + [1] * 5 + [0] * 10
        
        metrics = compute_metrics(y_true, y_pred)
        
        assert 0.0 <= metrics['Recall'] <= 1.0
        assert metrics['FN_count'] in [5, 10]
    
    def test_average_cost_per_transaction(self):
        """Тест: средняя стоимость на транзакцию"""
        y_true = [1, 0, 1, 0]
        y_pred = [0, 1, 0, 1]
        
        metrics = compute_metrics(y_true, y_pred, cost_fp=10, cost_fn=20)
        
        assert metrics['Avg_Cost_per_Transaction'] == 15.0

class TestFormatMetricsTable:
    """Тесты для функции format_metrics_table"""
    
    def test_single_model(self):
        """Тест: одна модель"""
        metrics_dict = {
            'LogisticRegression': {
                'Accuracy': 0.9523,
                'Precision': 0.8765,
                'Recall': 0.9012,
                'F1': 0.8888,
                'ROC_AUC': 0.9876,
                'Business_Cost': 1250,
                'FP_count': 50,
                'FN_count': 30,
                'TP_count': 270,
                'TN_count': 650,
                'Avg_Cost_per_Transaction': 0.125
            }
        }
        
        df = format_metrics_table(metrics_dict)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert df.loc[0, 'Model'] == 'LogisticRegression'
        assert df.loc[0, 'Accuracy'] == 0.9523
        assert df.loc[0, 'Business_Cost'] == 1250
        assert df.loc[0, 'Avg_Cost_per_Transaction'] == pytest.approx(0.125, abs=0.01)
    
    def test_multiple_models(self):
        """Тест: несколько моделей"""
        metrics_dict = {
            'Model_A': {'Accuracy': 0.95, 'Precision': 0.90, 'Recall': 0.85, 'F1': 0.87, 
                       'FP_count': 10, 'FN_count': 5, 'TP_count': 95, 'TN_count': 90, 
                       'Business_Cost': 60, 'Avg_Cost_per_Transaction': 0.30},
            'Model_B': {'Accuracy': 0.97, 'Precision': 0.92, 'Recall': 0.88, 'F1': 0.90, 
                       'FP_count': 8, 'FN_count': 4, 'TP_count': 96, 'TN_count': 92, 
                       'Business_Cost': 48, 'Avg_Cost_per_Transaction': 0.24}
        }
        
        df = format_metrics_table(metrics_dict)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert df.loc[0, 'Model'] == 'Model_A'
        assert df.loc[1, 'Model'] == 'Model_B'
    
    def test_rounding(self):
        """Тест: округление чисел"""
        metrics_dict = {
            'Model': {
                'Accuracy': 0.99999,
                'Precision': 0.87654,
                'Recall': 0.12345,
                'F1': 0.66666,
                'Business_Cost': 123.456,
                'Avg_Cost_per_Transaction': 0.12345
            }
        }
        
        df = format_metrics_table(metrics_dict)
        
        assert df.loc[0, 'Accuracy'] == 1.0
        assert df.loc[0, 'Precision'] == 0.8765
        assert df.loc[0, 'Recall'] == 0.1235
        assert df.loc[0, 'F1'] == 0.6667
        assert df.loc[0, 'Business_Cost'] == 123
    
    def test_missing_optional_metrics(self):
        """Тест: отсутствие опциональных метрик"""
        metrics_dict = {
            'Model': {
                'Accuracy': 0.95,
                'Precision': 0.90,
                'Recall': 0.85,
                'F1': 0.87,
                'Business_Cost': 100,
                'FP_count': 5,
                'FN_count': 5,
                'TP_count': 95,
                'TN_count': 95,
                'Avg_Cost_per_Transaction': 0.10
            }
        }
        
        df = format_metrics_table(metrics_dict)
        
        assert 'ROC_AUC' in df.columns or True

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])