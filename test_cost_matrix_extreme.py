import numpy as np
from sklearn.metrics import confusion_matrix

def calculate_business_cost(y_true, y_pred, fp_weight, fn_weight):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return fp * fp_weight + fn * fn_weight

print("=" * 60)
print("ПРОВЕРКА COST MATRIX НА КРАЙНИХ ЗНАЧЕНИЯХ")
print("=" * 60)

y_true = [1, 0, 1, 0, 1, 0]
y_pred_all_correct = [1, 0, 1, 0, 1, 0]
y_pred_with_fp = [1, 1, 1, 0, 1, 0]
y_pred_with_fn = [0, 0, 1, 0, 1, 0]
y_pred_with_both = [0, 1, 1, 0, 1, 0]

print("\n1. ТЕСТ 1: Все веса = 0")
cost = calculate_business_cost(y_true, y_pred_with_both, 0, 0)
print(f"   cost = {cost} (ожидается 0) -> {'ПРОЙДЕН' if cost == 0 else 'НЕ ПРОЙДЕН'}")

print("\n2. ТЕСТ 2: FP_weight = 1000, FN_weight = 0")
cost = calculate_business_cost(y_true, y_pred_with_fp, 1000, 0)
print(f"   cost = {cost} (ожидается 1000) -> {'ПРОЙДЕН' if cost == 1000 else 'НЕ ПРОЙДЕН'}")

print("\n3. ТЕСТ 3: FP_weight = 0, FN_weight = 1000")
cost = calculate_business_cost(y_true, y_pred_with_fn, 0, 1000)
print(f"   cost = {cost} (ожидается 1000) -> {'ПРОЙДЕН' if cost == 1000 else 'НЕ ПРОЙДЕН'}")

print("\n4. ТЕСТ 4: Оба веса = 1000")
cost = calculate_business_cost(y_true, y_pred_with_both, 1000, 1000)
print(f"   cost = {cost} (ожидается 2000) -> {'ПРОЙДЕН' if cost == 2000 else 'НЕ ПРОЙДЕН'}")

print("\n5. ТЕСТ 5: Идеальные предсказания")
cost = calculate_business_cost(y_true, y_pred_all_correct, 1000, 1000)
print(f"   cost = {cost} (ожидается 0) -> {'ПРОЙДЕН' if cost == 0 else 'НЕ ПРОЙДЕН'}")

print("\n6. ТЕСТ 6: Экстремальные веса (1_000_000)")
cost = calculate_business_cost(y_true, y_pred_with_both, 1000000, 1000000)
print(f"   cost = {cost} (ожидается 2000000) -> {'ПРОЙДЕН' if cost == 2000000 else 'НЕ ПРОЙДЕН'}")

print("\n7. ТЕСТ 7: Разные веса (100 и 1000)")
cost = calculate_business_cost(y_true, y_pred_with_both, 100, 1000)
print(f"   cost = {cost} (ожидается 1100) -> {'ПРОЙДЕН' if cost == 1100 else 'НЕ ПРОЙДЕН'}")

print("\n" + "=" * 60)
print("ВСЕ ТЕСТЫ ЗАВЕРШЕНЫ")
print("=" * 60)
