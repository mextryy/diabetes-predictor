import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import time
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    roc_auc_score, recall_score, precision_score, f1_score,
    accuracy_score, confusion_matrix, ConfusionMatrixDisplay,
)
from xgboost import XGBClassifier


# ── Ustawienia ────────────────────────────────────────────────────────────────
plt.rcParams["figure.dpi"] = 110
plt.rcParams["font.size"]  = 11
os.makedirs("charts", exist_ok=True)
os.makedirs("models", exist_ok=True)

K0   = "#2ecc71"
K1   = "#e74c3c"
PROG = 0.40

KOLORY_MODELI = {
    "XGBoost":              "#3498db",
    "Random Forest":        "#2ecc71",
    "Sieć neuronowa (MLP)": "#e74c3c",
}


# ── 1. Wczytanie i przygotowanie danych ───────────────────────────────────────
print("=" * 60)
print("  TRENOWANIE MODELI – DIABETES HEALTH INDICATORS")
print("  Problem: zdrowy (0) vs cukrzyca (2)")
print("  Przedcukrzyca (1) wykluczona z analizy")
print("=" * 60)

PLIK = "data/diabetes_012_health_indicators_BRFSS2021.csv"
if not os.path.exists(PLIK):
    print(f"\n[BŁĄD] Brak pliku: {PLIK}")
    exit()

df = pd.read_csv(PLIK)

# ZMIANA: usuwamy klasę 1 (przedcukrzyca), zostaje tylko 0 vs 2
df = df[df["Diabetes_012"] != 1].copy()
df["cel"] = (df["Diabetes_012"] == 2).astype(int)

print(f"\nLiczba rekordów po usunięciu przedcukrzycy: {len(df):,}")

CECHY = [c for c in df.columns if c not in ["Diabetes_012", "cel"]]


# ── 2. Balansowanie klas 50/50 przez próbkowanie ──────────────────────────────
# Zamiast RandomOverSampler (który duplikuje próbki) używamy undersamplingu:
# losowo dobieramy tyle samo zdrowych co chorych.
df_zdrowi = df[df["cel"] == 0]
df_chorzy = df[df["cel"] == 1]

print(f"\nRozkład przed balansowaniem:")
print(f"  Zdrowi:   {len(df_zdrowi):,}")
print(f"  Chorzy:   {len(df_chorzy):,}")

df_zdrowi_sample = df_zdrowi.sample(n=len(df_chorzy), random_state=42)
df_balanced = pd.concat([df_zdrowi_sample, df_chorzy])
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\nRozkład po balansowaniu 50/50:")
print(df_balanced["cel"].value_counts().to_string())


# ── 3. Podział na zbiory (70 / 15 / 15) ──────────────────────────────────────
X = df_balanced[CECHY].values.astype(np.float32)
y = df_balanced["cel"].values

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

print(f"\nPodział zakończony:")
print(f"  Trening:   {len(X_train):>7}  (70%)")
print(f"  Walidacja: {len(X_val):>7}  (15%)")
print(f"  Test:      {len(X_test):>7}  (15%)")


# ── 4. Skalowanie ─────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s   = scaler.transform(X_val)
X_test_s  = scaler.transform(X_test)
joblib.dump(scaler, "models/scaler.pkl")

print("\nSkalowanie zakończone.")


# ── 5. Definicja modeli ───────────────────────────────────────────────────────
# Uwaga: scale_pos_weight usunięty z XGBoost – dane są już zbalansowane 50/50.
# MLP: zamiast RandomOverSampler korzysta z tego samego zbalansowanego zbioru.
models = {
    "XGBoost": XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=150,
        max_depth=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),
    "Sieć neuronowa (MLP)": MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        alpha=0.01,
        max_iter=50,
        early_stopping=True,
        random_state=42
    ),
}


# ── 6. Trenowanie i ocena ─────────────────────────────────────────────────────
def ocen_model(model, X_te, y_te, prog=PROG):
    y_prob = model.predict_proba(X_te)[:, 1]
    y_pred = (y_prob >= prog).astype(int)

    return {
        "Accuracy":  round(accuracy_score(y_te, y_pred), 4),
        "AUC-ROC":   round(roc_auc_score(y_te, y_prob), 4),
        "Recall":    round(recall_score(y_te, y_pred), 4),
        "Precision": round(precision_score(y_te, y_pred), 4),
        "F1-Score":  round(f1_score(y_te, y_pred), 4),
    }, y_prob, y_pred


wyniki    = {}
wytrenowane = {}
predykcje   = {}

print("\n  TRENOWANIE MODELI...")

for nazwa, model in models.items():
    print(f"  > {nazwa}...", end=" ", flush=True)
    t0 = time.time()

    if nazwa == "XGBoost":
        # eval_set na zbiorze walidacyjnym – wczesne zatrzymanie
        model.fit(X_train_s, y_train, eval_set=[(X_val_s, y_val)], verbose=False)
    else:
        # Random Forest i MLP uczą się na zbalansowanym zbiorze treningowym
        model.fit(X_train_s, y_train)

    czas = time.time() - t0

    metryki, y_prob, y_pred = ocen_model(model, X_test_s, y_test)

    wyniki[nazwa]      = metryki
    wytrenowane[nazwa] = model
    predykcje[nazwa]   = (y_prob, y_pred)

    print(f"OK ({czas:.1f}s)")


# ── 7. Zapis modeli ───────────────────────────────────────────────────────────
for nazwa, model in wytrenowane.items():
    nazwa_pliku = nazwa.lower().replace(" ", "_").replace("(", "").replace(")", "")
    joblib.dump(model, f"models/{nazwa_pliku}.pkl")

joblib.dump(CECHY, "models/cechy.pkl")
print("\nModele zapisane do models/")


# ── 8. Tabela wyników ─────────────────────────────────────────────────────────
df_wyniki = pd.DataFrame(wyniki).T
print("\n" + "=" * 60)
print(df_wyniki.to_string())
print("=" * 60)
joblib.dump(df_wyniki, "models/wyniki_porownanie.csv")
df_wyniki.to_csv("models/wyniki_porownanie.csv")


# ── 9. Wykres porównawczy ─────────────────────────────────────────────────────
metryki_listy = ["Accuracy", "AUC-ROC", "Recall", "Precision", "F1-Score"]
nazwy_m = list(wyniki.keys())

fig, axes = plt.subplots(1, 5, figsize=(20, 5))
fig.suptitle(
    f"Porównanie modeli – zdrowy vs cukrzyca (próg={PROG})",
    fontweight="bold", fontsize=13
)

for ax, metryka in zip(axes, metryki_listy):
    wartosci = [wyniki[n][metryka] for n in nazwy_m]
    labels_short = ["XGB", "RF", "MLP"]
    bars = ax.bar(
        labels_short, wartosci,
        color=list(KOLORY_MODELI.values()), alpha=0.85
    )
    for bar, val in zip(bars, wartosci):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.2f}", ha="center", fontsize=9
        )
    ax.set_title(metryka, fontweight="bold")
    ax.set_ylim(0, 1.1)

plt.tight_layout()
plt.savefig("charts/08_porownanie_modeli.png")
plt.show()


# ── 10. Macierze błędów ───────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
fig.suptitle("Macierze błędów – zdrowy vs cukrzyca", fontweight="bold")

for ax, (nazwa, (_, y_pred)) in zip(axes, predykcje.items()):
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(
        cm, display_labels=["Zdrowy", "Cukrzyca"]
    ).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(nazwa)

plt.tight_layout()
plt.savefig("charts/11_macierze_bledow.png")
plt.show()

print("\n✓ Modele i wykresy zapisane. Możesz przejść do kroku 3!")