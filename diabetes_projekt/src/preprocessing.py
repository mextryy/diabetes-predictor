import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ── Ustawienia ────────────────────────────────────────────────────────────────
plt.rcParams["figure.dpi"] = 110
plt.rcParams["font.size"]  = 11
os.makedirs("charts", exist_ok=True)

K0 = "#2ecc71"   # zielony       – brak cukrzycy
K1 = "#f39c12"   # pomarańczowy  – przedcukrzyca
K2 = "#e74c3c"   # czerwony      – cukrzyca

OPISY = {
    "Diabetes_012":         "Cukrzyca (0=brak, 1=przedcukrzyca, 2=cukrzyca)",
    "HighBP":               "Nadciśnienie tętnicze",
    "HighChol":             "Wysoki cholesterol",
    "CholCheck":            "Badanie cholesterolu w ost. 5 latach",
    "BMI":                  "Wskaźnik masy ciała (BMI)",
    "Smoker":               "Palenie tytoniu (>100 papier. w życiu)",
    "Stroke":               "Przebyty udar mózgu",
    "HeartDiseaseorAttack": "Choroba serca / zawał",
    "PhysActivity":         "Aktywność fizyczna (ost. 30 dni)",
    "Fruits":               "Owoce ≥1 raz dziennie",
    "Veggies":              "Warzywa ≥1 raz dziennie",
    "HvyAlcoholConsump":    "Ciężkie spożycie alkoholu",
    "AnyHealthcare":        "Ubezpieczenie zdrowotne",
    "NoDocbcCost":          "Brak wizyty lekarskiej z powodu kosztów",
    "GenHlth":              "Ogólny stan zdrowia (1=doskonały, 5=zły)",
    "MentHlth":             "Zły stan psychiczny (liczba dni/miesiąc)",
    "PhysHlth":             "Zły stan fizyczny (liczba dni/miesiąc)",
    "DiffWalk":             "Trudności z chodzeniem",
    "Sex":                  "Płeć (0=kobieta, 1=mężczyzna)",
    "Age":                  "Wiek (1=18-24 lat, 13=≥80 lat)",
    "Education":            "Wykształcenie (1=brak, 6=wyższe)",
    "Income":               "Dochód (1=<$10k, 8=>$75k)",
}


# ── Wczytanie danych ──────────────────────────────────────────────────────────
print("=" * 60)
print("  ANALIZA ZBIORU DANYCH – DIABETES HEALTH INDICATORS")
print("=" * 60)

PLIK = "diabetes_projekt/data/diabetes_012_health_indicators_BRFSS2021.csv"

if not os.path.exists(PLIK):
    print(f"\n[BŁĄD] Nie znaleziono pliku: {PLIK}")
    print("\nPobierz dataset z Kaggle:")
    print("  https://www.kaggle.com/code/ahmedatalla/diabetes-health-indicators/input")
    print("  → pobierz plik: diabetes_012_health_indicators_BRFSS2021.csv")
    print("  → wklej go do folderu 'data/'")
    exit()

df = pd.read_csv(PLIK)
print(f"\n✓ Wczytano {len(df):,} rekordów, {df.shape[1]} kolumn")


# ── Podstawowe informacje ─────────────────────────────────────────────────────
print("\n--- Pierwsze 5 wierszy ---")
print(df.head().to_string())

print("\n--- Typy kolumn i brakujące wartości ---")
info = pd.DataFrame({
    "typ":     df.dtypes,
    "brakuje": df.isnull().sum(),
    "min":     df.min(),
    "max":     df.max(),
})
print(info.to_string())

print("\n--- Statystyki opisowe ---")
print(df.describe().round(2).to_string())


# ── Rozkład klasy docelowej ───────────────────────────────────────────────────
print("\n--- Rozkład klasy docelowej (Diabetes_012) ---")
counts = df["Diabetes_012"].value_counts().sort_index()
etykiety = {0: "Brak cukrzycy (0)", 1: "Przedcukrzyca (1)", 2: "Cukrzyca (2)"}
for cls, cnt in counts.items():
    print(f"  {etykiety[cls]:25s}: {cnt:7,}  ({cnt/len(df)*100:.1f}%)")

# Uwaga dla czytelnika
print("\n  UWAGA: Poniższe wykresy (EDA) przedstawiają pełny zbiór danych")
print("  ze wszystkimi 3 klasami. W etapie modelowania (train.py,")
print("  optimize_xgboost.py) przedcukrzyca (klasa 1) jest wykluczona –")
print("  model klasyfikuje wyłącznie: zdrowy (0) vs cukrzyca (2).")


# ── WYKRES 1 – Rozkład klas ───────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Rozkład klasy docelowej: Diabetes_012", fontweight="bold", fontsize=13)

kolory = [K0, K1, K2]
nazwy  = ["Brak cukrzycy\n(0)", "Przedcukrzyca\n(1)", "Cukrzyca\n(2)"]

ax1.pie(counts, labels=nazwy, colors=kolory, autopct="%1.1f%%",
        startangle=90, wedgeprops={"edgecolor": "white", "linewidth": 2})
ax1.set_title("Proporcje klas")

bars = ax2.bar(nazwy, counts, color=kolory, edgecolor="white", linewidth=1.5)
for bar, val in zip(bars, counts):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 500,
             f"{val:,}", ha="center", fontweight="bold", fontsize=11)
ax2.set_ylabel("Liczba respondentów")
ax2.set_title("Liczba rekordów per klasa")

plt.tight_layout()
plt.savefig("charts/01_rozklad_klas.png", bbox_inches="tight")
plt.show()
print("\nZapisano: charts/01_rozklad_klas.png")


# ── WYKRES 2 – Wiek a cukrzyca ────────────────────────────────────────────────
etykiety_wieku = ["18-24","25-29","30-34","35-39","40-44",
                  "45-49","50-54","55-59","60-64","65-69",
                  "70-74","75-79","80+"]

ryzyko_wiek = df.groupby("Age")["Diabetes_012"].apply(
    lambda x: (x == 2).mean() * 100
)

fig, ax = plt.subplots(figsize=(12, 5))
bars = ax.bar(range(1, 14), ryzyko_wiek.values,
              color=K2, alpha=0.85, edgecolor="white")
ax.set_xticks(range(1, 14))
ax.set_xticklabels(etykiety_wieku, rotation=30)
ax.axhline(
    df["Diabetes_012"].eq(2).mean() * 100,
    color="orange", linestyle="--", linewidth=2,
    label=f"Średnia: {df['Diabetes_012'].eq(2).mean()*100:.1f}%"
)
for bar, val in zip(bars, ryzyko_wiek.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
            f"{val:.1f}%", ha="center", fontsize=8, fontweight="bold")
ax.set_title("Odsetek przypadków cukrzycy wg grupy wiekowej",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Grupa wiekowa")
ax.set_ylabel("% z cukrzycą")
ax.legend()
plt.tight_layout()
plt.savefig("charts/02_wiek_a_cukrzyca.png", bbox_inches="tight")
plt.show()
print("Zapisano: charts/02_wiek_a_cukrzyca.png")


# ── WYKRES 3 – BMI: rozkład per klasa ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
for cls, kolor, etykieta in [
    (0, K0, "Brak cukrzycy"),
    (1, K1, "Przedcukrzyca"),
    (2, K2, "Cukrzyca")
]:
    dane = df[df["Diabetes_012"] == cls]["BMI"]
    ax.hist(dane, bins=50, alpha=0.5, color=kolor,
            label=f"{etykieta} (śr.={dane.mean():.1f})", density=True)

ax.axvline(25, color="gray",  linestyle="--", linewidth=1.5, label="Nadwaga (BMI=25)")
ax.axvline(30, color="black", linestyle="--", linewidth=1.5, label="Otyłość (BMI=30)")
ax.set_title("Rozkład BMI wg klasy cukrzycy", fontsize=13, fontweight="bold")
ax.set_xlabel("BMI")
ax.set_ylabel("Gęstość")
ax.set_xlim(10, 70)
ax.legend()
plt.tight_layout()
plt.savefig("charts/03_bmi_rozklad.png", bbox_inches="tight")
plt.show()
print("Zapisano: charts/03_bmi_rozklad.png")


# ── WYKRES 4 – Kluczowe czynniki ryzyka ──────────────────────────────────────
cechy_binarne = ["HighBP", "HighChol", "Smoker", "Stroke",
                 "HeartDiseaseorAttack", "PhysActivity",
                 "Fruits", "Veggies", "HvyAlcoholConsump", "DiffWalk"]

wyniki = {}
for cecha in cechy_binarne:
    ma_ceche = df[df[cecha] == 1]["Diabetes_012"].eq(2).mean() * 100
    nie_ma   = df[df[cecha] == 0]["Diabetes_012"].eq(2).mean() * 100
    wyniki[OPISY.get(cecha, cecha)] = (ma_ceche, nie_ma)

fig, ax = plt.subplots(figsize=(14, 6))
y_pos = np.arange(len(wyniki))
nazwy_c      = list(wyniki.keys())
wartosci_ma  = [v[0] for v in wyniki.values()]
wartosci_nie = [v[1] for v in wyniki.values()]

ax.barh(y_pos - 0.2, wartosci_ma,  height=0.38, color=K2, alpha=0.85, label="Ma cechę")
ax.barh(y_pos + 0.2, wartosci_nie, height=0.38, color=K0, alpha=0.85, label="Nie ma cechy")
ax.set_yticks(y_pos)
ax.set_yticklabels(nazwy_c, fontsize=9)
ax.set_xlabel("% z cukrzycą")
ax.set_title("Odsetek cukrzycy przy obecności/braku czynnika ryzyka",
             fontsize=12, fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig("charts/04_czynniki_ryzyka.png", bbox_inches="tight")
plt.show()
print("✓ Zapisano: charts/04_czynniki_ryzyka.png")


# ── WYKRES 5 – Stan zdrowia (GenHlth) a cukrzyca ─────────────────────────────
etykiety_hlth = {1: "Doskonały", 2: "Bardzo dobry", 3: "Dobry",
                 4: "Dostateczny", 5: "Zły"}
ryzyko_hlth = df.groupby("GenHlth")["Diabetes_012"].apply(
    lambda x: (x == 2).mean() * 100
)

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(
    [etykiety_hlth[i] for i in ryzyko_hlth.index],
    ryzyko_hlth.values,
    color=K2, alpha=0.85, edgecolor="white"
)
for bar, val in zip(bars, ryzyko_hlth.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
            f"{val:.1f}%", ha="center", fontweight="bold")
ax.set_title("Odsetek cukrzycy wg ogólnego stanu zdrowia (GenHlth)",
             fontsize=12, fontweight="bold")
ax.set_xlabel("Ogólny stan zdrowia")
ax.set_ylabel("% z cukrzycą")
plt.tight_layout()
plt.savefig("charts/05_stan_zdrowia.png", bbox_inches="tight")
plt.show()
print("✓ Zapisano: charts/05_stan_zdrowia.png")


# ── WYKRES 6 – Mapa korelacji ─────────────────────────────────────────────────
kolumny_korelacji = ["Diabetes_012", "BMI", "Age", "GenHlth", "HighBP",
                     "HighChol", "HeartDiseaseorAttack", "Stroke",
                     "PhysActivity", "DiffWalk", "Income", "Education"]

fig, ax = plt.subplots(figsize=(11, 9))
maska = np.triu(np.ones(len(kolumny_korelacji), dtype=bool))
sns.heatmap(
    df[kolumny_korelacji].corr(),
    mask=maska,
    annot=True, fmt=".2f",
    cmap="RdYlBu_r", vmin=-1, vmax=1,
    ax=ax, linewidths=0.5,
    annot_kws={"size": 9}
)
ax.set_title("Macierz korelacji wybranych zmiennych z Diabetes_012",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig("charts/06_korelacje.png", bbox_inches="tight")
plt.show()
print("✓ Zapisano: charts/06_korelacje.png")


# ── WYKRES 7 – Dochód i wykształcenie a cukrzyca ─────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Czynniki społeczno-ekonomiczne a cukrzyca",
             fontweight="bold", fontsize=13)

ryzyko_income = df.groupby("Income")["Diabetes_012"].apply(
    lambda x: (x == 2).mean() * 100
)
ax1.plot(ryzyko_income.index, ryzyko_income.values,
         color=K2, marker="o", linewidth=2.5, markersize=8)
ax1.fill_between(ryzyko_income.index, ryzyko_income.values, alpha=0.2, color=K2)
ax1.set_xticks(range(1, 9))
ax1.set_xticklabels(
    ["<$10k","$10-15k","$15-20k","$20-25k","$25-35k","$35-50k","$50-75k",">$75k"],
    rotation=30
)
ax1.set_title("Dochód a cukrzyca")
ax1.set_ylabel("% z cukrzycą")
ax1.set_xlabel("Przedział dochodowy")

ryzyko_edu = df.groupby("Education")["Diabetes_012"].apply(
    lambda x: (x == 2).mean() * 100
)
etyk_edu = {1: "Brak", 2: "Podstawowe", 3: "Niepełne śr.",
            4: "Średnie", 5: "Niepełne wyższe", 6: "Wyższe"}
ax2.bar(
    [etyk_edu.get(i, str(i)) for i in ryzyko_edu.index],
    ryzyko_edu.values,
    color=K2, alpha=0.85, edgecolor="white"
)
ax2.set_title("Wykształcenie a cukrzyca")
ax2.set_ylabel("% z cukrzycą")
ax2.tick_params(axis="x", rotation=30)

plt.tight_layout()
plt.savefig("charts/07_spoleczno_ekonomiczne.png", bbox_inches="tight")
plt.show()
print("✓ Zapisano: charts/07_spoleczno_ekonomiczne.png")


# ── Podsumowanie ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  PODSUMOWANIE ANALIZY")
print("=" * 60)
print(f"  Rekordów łącznie:    {len(df):,}")
print(f"  Brak cukrzycy (0):   {counts[0]:,}  ({counts[0]/len(df)*100:.1f}%)")
if 1 in counts:
    print(f"  Przedcukrzyca (1):   {counts[1]:,}  ({counts[1]/len(df)*100:.1f}%)")
print(f"  Cukrzyca (2):        {counts[2]:,}  ({counts[2]/len(df)*100:.1f}%)")
print(f"  Brakujące wartości:  {df.isnull().sum().sum()} (brak!)")
print(f"  Wykresy zapisane w:  charts/")
print(f"\n  MODELOWANIE: klasa 1 (przedcukrzyca) wykluczona.")
print(f"  Model klasyfikuje: zdrowy (0) vs cukrzyca (2).")
print(f"\n  Następny krok: python src/train.py")
print("=" * 60)