
import joblib
import numpy as np
import os
import sys

try:
    from xgboost import XGBClassifier
except ImportError:
    print("\n[BŁĄD] Brak biblioteki xgboost. Zainstaluj ją: pip install xgboost")
    sys.exit()


FOLDER = "models"
PROG = 0.40


def zaladuj():
    wymagane = [
        "xgboost_optymalizowany.pkl",
        "scaler.pkl",
        "cechy.pkl"
    ]

    brak = [f for f in wymagane if not os.path.exists(f"{FOLDER}/{f}")]

    if brak:
        print(f"\n[BŁĄD] Nie znaleziono plików: {brak}")
        sys.exit()

    model  = joblib.load(f"{FOLDER}/xgboost_optymalizowany.pkl")
    scaler = joblib.load(f"{FOLDER}/scaler.pkl")
    cechy  = joblib.load(f"{FOLDER}/cechy.pkl")

    return model, scaler, cechy


def pytaj_tak_nie(pytanie):
    while True:
        odp = input(f"  {pytanie} [t/n]: ").strip().lower()

        if odp in ("t", "tak", "y", "yes", "1"):
            return 1

        if odp in ("n", "nie", "no", "0"):
            return 0

        print("    Wpisz 't' lub 'n'")


def pytaj_liczbe(pytanie, min_val, max_val):
    while True:
        try:
            val = float(input(f"  {pytanie}: ").strip())

            if min_val <= val <= max_val:
                return val

            print(f"    Podaj wartość od {min_val} do {max_val}")

        except ValueError:
            print("    Wpisz liczbę")


def pytaj_wybor(pytanie, opcje_dict):
    print(f"\n  {pytanie}")

    for klucz, opis in opcje_dict.items():
        print(f"    {klucz}. {opis}")

    while True:
        try:
            wybor = int(input("    Twój wybór (numer): ").strip())

            if wybor in opcje_dict:
                return wybor

            print(f"    Wybierz numer od {min(opcje_dict)} do {max(opcje_dict)}")

        except ValueError:
            print("    Wpisz numer")


def zbierz_dane():
    print("\n  ANKIETA MEDYCZNA - PODAJ DANE DO ANALIZY AI\n")

    d = {}

    # Stan zdrowia
    d["HighBP"]               = pytaj_tak_nie("Czy masz nadciśnienie tętnicze?")
    d["HighChol"]             = pytaj_tak_nie("Czy masz wysoki cholesterol?")
    d["CholCheck"]            = pytaj_tak_nie("Czy badałeś cholesterol w ciągu ostatnich 5 lat?")
    d["Stroke"]               = pytaj_tak_nie("Czy kiedykolwiek miałeś udar mózgu?")
    d["HeartDiseaseorAttack"] = pytaj_tak_nie("Czy masz chorobę wieńcową lub miałeś zawał?")
    d["DiffWalk"]             = pytaj_tak_nie("Czy masz poważne trudności z chodzeniem lub wchodzeniem po schodach ?")

    # BMI
    d["BMI"] = pytaj_liczbe("Twoje BMI (waga[kg] / wzrost[m]^2)", 10, 80)

    # Styl życia
    d["Smoker"]            = pytaj_tak_nie("Czy wypaliłeś w życiu min. 100 papierosów (5 paczek)?")
    d["HvyAlcoholConsump"] = pytaj_tak_nie("Czy nadużywasz alkoholu?")
    d["PhysActivity"]      = pytaj_tak_nie("Czy ćwiczyłeś w ciągu ostatnich 30 dni?")
    d["Fruits"]            = pytaj_tak_nie("Czy jesz owoce co najmniej raz dziennie?")
    d["Veggies"]           = pytaj_tak_nie("Czy jesz warzywa co najmniej raz dziennie?")

    # Opieka medyczna
    d["AnyHealthcare"] = pytaj_tak_nie("Czy masz ubezpieczenie / stałą opiekę medyczną?")
    d["NoDocbcCost"]   = pytaj_tak_nie("Czy zrezygnowałeś z lekarza w ost. roku przez koszty?")

    d["GenHlth"] = pytaj_wybor(
        "Ogólny stan zdrowia:",
        {
            1: "Doskonałe",
            2: "Bardzo dobre",
            3: "Dobre",
            4: "Dostateczne",
            5: "Złe"
        }
    )

    d["MentHlth"] = int(pytaj_liczbe("Ile dni w miesiącu czułeś się źle psychicznie (0-30)?", 0, 30))
    d["PhysHlth"] = int(pytaj_liczbe("Ile dni w miesiącu czułeś się źle fizycznie (0-30)?", 0, 30))

    # Dane demograficzne
    d["Sex"] = pytaj_wybor(
        "Płeć:",
        {0: "Kobieta", 1: "Mężczyzna"}
    )

    d["Age"] = pytaj_wybor(
        "Wiek (kategorie):",
        {
            1:  "18-24 lata",
            2:  "25-29 lat",
            3:  "30-34 lata",
            4:  "35-39 lat",
            5:  "40-44 lata",
            6:  "45-49 lat",
            7:  "50-54 lata",
            8:  "55-59 lat",
            9:  "60-64 lata",
            10: "65-69 lat",
            11: "70-74 lata",
            12: "75-79 lat",
            13: "80+"
        }
    )

    d["Education"] = pytaj_wybor(
        "Wykształcenie:",
        {
            1: "Brak",
            2: "Szkoła podstawowa",
            3: "Gimnazjum / brak matury",
            4: "Szkoła średnia / matura",
            5: "Studia niepełne / policealna",
            6: "Studia wyższe / magister"
        }
    )

    d["Income"] = pytaj_wybor(
        "Dochód roczny (PLN):",
    {
        1: "< 40 tys. zł",
        2: "40 tys. - 60 tys. zł",
        3: "60 tys. - 80 tys. zł",
        4: "80 tys. - 100 tys. zł",
        5: "100 tys. - 140 tys. zł",
        6: "140 tys. - 200 tys. zł",
        7: "200 tys. - 300 tys. zł",
        8: "> 300 tys. zł"
    }
    )

    return d


def main():
    print("  KALKULATOR RYZYKA CUKRZYCY ")
    print("  Model:XGBOOST")
    print(f"  Próg decyzyjny = {PROG}")
    print("\n  Uwaga: kalkulator ocenia ryzyko cukrzycy ale nie zastępuje konsultacji lekarskiej\n")
 

    model, scaler, CECHY = zaladuj()

    while True:
        dane = zbierz_dane()

        X_wektor   = np.array([[dane[c] for c in CECHY]], dtype=np.float32)
        X_skalowane = scaler.transform(X_wektor)

        p = float(model.predict_proba(X_skalowane)[0, 1])

        print("\n" + "═" * 60)
        print("  WYNIK ANALIZY – MODEL XGBOOST")
        print("═" * 60)

        wynik = "PODWYŻSZONE RYZYKO CUKRZYCY" if p >= PROG else "NISKIE RYZYKO CUKRZYCY"

        print(f"\n  Prawdopodobieństwo cukrzycy: {p * 100:.1f}%")
        print(f"  Próg decyzyjny:              {PROG}")
        print(f"  Wynik modelu:                {wynik}")
        print("-" * 60)

        if p >= 0.70:
            print("\n   !ALARM: Bardzo wysokie prawdopodobieństwo.!")
            print("     Skonsultuj się z lekarzem i wykonaj badanie glukozy!")

        elif p >= PROG:
            print("\n   OSTRZEŻENIE: Ryzyko jest istotne.")
            print("     Rozważ badania krwi i zmianę stylu życia.")

        else:
            print("\n   INFO: Model nie wykazuje wysokiego ryzyka cukrzycy.")
            print("     Dbaj o zdrowy styl życia i regularne badania.")

        dalej = input("\n  Sprawdzić inne dane? [t/n]: ").strip().lower()

        if dalej not in ("t", "tak", "y", "yes", "1"):
            break

    print("\n  Do widzenia!\n")


if __name__ == "__main__":
    main()