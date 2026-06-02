#  Diabetes Predictor 

System wspomagania decyzji medycznych oparty na sztucznej inteligencji, służący do szacowania ryzyka wystąpienia cukrzycy. Projekt realizowany w ramach przedmiotu **Podstawy Sztucznej Inteligencji (PSI)**.

##  Cechy projektu
* **Podział danych:** Rygorystyczny podział na trzy zbiory: **70% treningowy**, **15% walidacyjny** (do monitorowania procesu nauki) oraz **15% testowy** (do ostatecznej oceny). Podział zachowuje idealną stratyfikację klas.
* **Strategia balansowania danych:** Zastosowanie techniki **Random Undersampling** (podpróbkowania klasy większościowej) zamiast oversamplingu. Usunięcie nadmiaru danych osób zdrowych zmusza modele do lepszego rozpoznawania wzorców chorobowych i drastycznie podnosi metrykę **Precision**.
* **Trzy modele AI:** Wdrożenie i pełne porównanie trzech różnych architektur:
  * **XGBoost** (ze zbalansowaną wagą klas `scale_pos_weight`)
  * **Random Forest** (z parametrem `class_weight="balanced"`)
  * **Sieć Neuronowa MLP** (Multi-layer Perceptron o strukturze warstw ukrytych 64x32)
* **Optymalizacja medyczna:** Próg decyzyjny został dostrojony do poziomu **0.80**, co pozwoliło ustabilizować precyzję na poziomie ~50% przy jednoczesnym zachowaniu bardzo wysokiej czułości (**Recall ~85%**), kluczowej w diagnostyce przesiewowej.
* **Adaptacja lokalna:** Zmienne socjoekonomiczne (dochód roczny) zostały zaadaptowane kulturowo z oryginalnych przedziałów USD na wartości naturalne dla polskiego rynku (PLN).

##  Struktura plików projektu
* `src/train.py` - Główny skrypt preprocessingu, trenowania modeli z uwzględnieniem undersamplingu oraz generowania pełnych wykresów jakości.
* `src/predict.py` - Interaktywny, konsolowy kalkulator ryzyka pozwalający na wprowadzenie własnych danych ankietowych w celach testowych.
* `charts/` - Folder zawierający wygenerowane macierze błędów (`11_macierze_bledow.png`), wykresy słupkowe metryk (`08_porownanie_modeli.png`), krzywe ROC i PR oraz historię strat sieci MLP.
* `models/` - Zapisane obiekty modeli (`.pkl`), struktura cech oraz wytrenowany skaler danych.

##  Instrukcja uruchomienia

### 1. Instalacja środowiska
Upewnij się, że masz zainstalowanego Pythona (zalecany 3.10+). Zainstaluj wymagane pakiety za pomocą pliku wymagań:
```bash
pip install -r requirements.txt
```

### 2. Przygotowanie danych
Umieść plik ze zbiorem danych o nazwie diabetes_prediction_dataset.csv w głównym katalogu projektu (obok folderu src).

### 3. Trenowanie i generowanie raportów
Uruchom proces uczenia modeli od nowa, aby wygenerować świeże pliki .pkl oraz zaktualizować wykresy w folderze charts:

```
python src/train.py
```
### 4. Uruchomienie kalkulatora diagnostycznego
Aby przetestować działanie modeli w praktyce i sprawdzić indywidualne ryzyko na podstawie pytań ankietowych (z przedziałami dochodowymi w PLN), uruchom:

```
python src/predict.py
```
## Uzyskane wyniki (Zbiór Testowy)
Po zastosowaniu metody Undersamplingu oraz podniesieniu progu do 0.80, modele przestały generować masowe fałszywe alarmy:

Recall (~85%): Gwarantuje wykrycie zdecydowanej większości chorych pacjentów.

Precision (~50%): Co druga osoba wskazana przez system jako "zagrożona" faktycznie cierpi na cukrzycę, co jest wynikiem wysoce satysfakcjonującym w medycznych testach przesiewowych.
