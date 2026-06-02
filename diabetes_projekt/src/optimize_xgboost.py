import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    recall_score,
    precision_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
)
from xgboost import XGBClassifier


# ── Ustawienia ─────────────────────────────────────────
os.makedirs("models", exist_ok=True)
os.makedirs("charts", exist_ok=True)

PLIK = "diabetes_projekt/data/diabetes_012_health_indicators_BRFSS2021.csv"
FINALNY_PROG = 0.40


# ── 1. Wczytanie danych ────────────────────────────────
df = pd.read_csv(PLIK)

df = df[df["Diabetes_012"] != 1].copy()
df["cel"] = (df["Diabetes_012"] == 2).astype(int)

print("=" * 60)
print("  OPTYMALIZACJA MODELU XGBOOST")
print("  Problem: zdrowy (0) vs cukrzyca (2)")
print("  Przedcukrzyca (1) wykluczona z analizy")
print("=" * 60)

print(f"\nLiczba rekordów po usunięciu przedcukrzycy: {len(df)}")

CECHY = [c for c in df.columns if c not in ["Diabetes_012", "cel"]]


# ── 2. Balansowanie klas 50/50 ─────────────────────────
df_zdrowi = df[df["cel"] == 0]
df_chorzy = df[df["cel"] == 1]

print(f"\nRozkład przed balansowaniem:")
print(f"  Zdrowi:   {len(df_zdrowi)}")
print(f"  Chorzy:   {len(df_chorzy)}")

df_zdrowi_sample = df_zdrowi.sample(n=len(df_chorzy), random_state=42)

df_balanced = pd.concat([df_zdrowi_sample, df_chorzy])
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\nRozkład po balansowaniu 50/50:")
print(df_balanced["cel"].value_counts())


# ── 3. Podział danych ──────────────────────────────────
X = df_balanced[CECHY].values.astype(np.float32)
y = df_balanced["cel"].values

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

print("\nPodział danych:")
print(f"  Trening:   {len(X_train)}")
print(f"  Walidacja: {len(X_val)}")
print(f"  Test:      {len(X_test)}")


# ── 4. Skalowanie ──────────────────────────────────────
scaler = StandardScaler()

X_train_s = scaler.fit_transform(X_train)
X_val_s   = scaler.transform(X_val)
X_test_s  = scaler.transform(X_test)


# ── 5. Strojenie hiperparametrów XGBoost ───────────────
param_grid = {
    "n_estimators":     [100, 200],
    "max_depth":        [3, 4, 5],
    "learning_rate":    [0.05, 0.1],
    "subsample":        [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
    "min_child_weight": [1, 5],
    "gamma":            [0, 0.1],
}

xgb_base = XGBClassifier(
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)

grid = GridSearchCV(
    estimator=xgb_base,
    param_grid=param_grid,
    scoring="roc_auc",
    cv=3,
    n_jobs=-1,
    verbose=1
)

print("\nRozpoczynam strojenie XGBoost...")
grid.fit(X_train_s, y_train)

print("\nNajlepsze parametry:")
print(grid.best_params_)


# ── 6. Early stopping na zbiorze walidacyjnym ──────────
best_params = grid.best_params_.copy()
best_params.pop("n_estimators")

best_model = XGBClassifier(
    **best_params,
    n_estimators=1000,
    early_stopping_rounds=20,
    eval_metric="auc",
    random_state=42,
    n_jobs=-1
)

best_model.fit(
    X_train_s,
    y_train,
    eval_set=[(X_val_s, y_val)],
    verbose=False
)

print(f"\nEarly stopping: najlepsza liczba drzew = {best_model.best_iteration}")

y_prob_val = best_model.predict_proba(X_val_s)[:, 1]
auc_val = roc_auc_score(y_val, y_prob_val)
print(f"AUC-ROC na walidacji: {auc_val:.4f}")


# ── 7. Test różnych progów ─────────────────────────────
progi = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
wyniki = []

y_prob_test = best_model.predict_proba(X_test_s)[:, 1]

for prog in progi:
    y_pred = (y_prob_test >= prog).astype(int)

    wyniki.append({
        "Próg":      prog,
        "Accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "AUC-ROC":   round(roc_auc_score(y_test, y_prob_test), 4),
        "Recall":    round(recall_score(y_test, y_pred), 4),
        "Precision": round(precision_score(y_test, y_pred), 4),
        "F1-Score":  round(f1_score(y_test, y_pred), 4)
    })

df_wyniki = pd.DataFrame(wyniki)

print("\nWyniki dla różnych progów (zbiór testowy):")
print(df_wyniki.to_string(index=False))

df_wyniki.to_csv("models/wyniki_xgboost_optymalizacja.csv", index=False)


# ── 8. Finalny próg ────────────────────────────────────
y_pred_final = (y_prob_test >= FINALNY_PROG).astype(int)

print(f"\nFinalny wybrany próg decyzyjny: {FINALNY_PROG}")

metryki_finalne = {
    "Accuracy":  round(accuracy_score(y_test, y_pred_final), 4),
    "AUC-ROC":   round(roc_auc_score(y_test, y_prob_test), 4),
    "Recall":    round(recall_score(y_test, y_pred_final), 4),
    "Precision": round(precision_score(y_test, y_pred_final), 4),
    "F1-Score":  round(f1_score(y_test, y_pred_final), 4)
}

print("\nMetryki finalnego modelu (zbiór testowy):")
for nazwa, wartosc in metryki_finalne.items():
    print(f"  {nazwa}: {wartosc}")


# ── 9. Macierz błędów ──────────────────────────────────
cm = confusion_matrix(y_test, y_pred_final)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Zdrowy", "Cukrzyca"]
)

disp.plot(cmap="Blues")
plt.title(f"XGBoost: zdrowy vs cukrzyca, próg={FINALNY_PROG}")
plt.tight_layout()
plt.savefig("charts/12_xgboost_optymalizacja_macierz.png")
plt.show()


# ── 10. Wykres porównania progów ───────────────────────
df_wyniki.plot(
    x="Próg",
    y=["Recall", "Precision", "F1-Score"],
    marker="o"
)

plt.title("Wpływ progu decyzyjnego na metryki XGBoost")
plt.ylabel("Wartość metryki")
plt.ylim(0, 1)
plt.grid(True)
plt.tight_layout()
plt.savefig("charts/13_xgboost_progi.png")
plt.show()


# ── 11. Krzywa ROC ─────────────────────────────────────
fpr, tpr, thresholds = roc_curve(y_test, y_prob_test)
auc_test = roc_auc_score(y_test, y_prob_test)

idx = np.argmin(np.abs(thresholds - FINALNY_PROG))

plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, color="steelblue", lw=2, label=f"XGBoost (AUC = {auc_test:.4f})")
plt.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Losowy klasyfikator")
plt.scatter(
    fpr[idx], tpr[idx],
    color="red", zorder=5,
    label=f"Próg = {FINALNY_PROG} (TPR={tpr[idx]:.2f}, FPR={fpr[idx]:.2f})"
)
plt.xlabel("False Positive Rate (1 - Specificity)")
plt.ylabel("True Positive Rate (Recall)")
plt.title("Krzywa ROC – XGBoost: zdrowy vs cukrzyca")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.savefig("charts/14_xgboost_roc.png")
plt.show()


# ── 12. Zapis finalnego modelu ─────────────────────────
joblib.dump(best_model,   "models/xgboost_optymalizowany.pkl")
joblib.dump(scaler,       "models/scaler.pkl")
joblib.dump(CECHY,        "models/cechy.pkl")
joblib.dump(FINALNY_PROG, "models/prog_xgboost.pkl")

print("\nZapisano:")
print("- models/xgboost_optymalizowany.pkl")
print("- models/scaler.pkl")
print("- models/cechy.pkl")
print("- models/prog_xgboost.pkl")
print("- models/wyniki_xgboost_optymalizacja.csv")
print("- charts/12_xgboost_optymalizacja_macierz.png")
print("- charts/13_xgboost_progi.png")
print("- charts/14_xgboost_roc.png")