### a) Co robi StandardScaler?

Standard Scaler to klasa, która służy do standaryzacji danych, w taki sposób aby dane wyjściowe miały średnią równą 0 i odchylenie standardowe równe 1. Wzór, który jest stosowany do przeprowadzenia standaryzacji:

$$z = \dfrac{x - mean}{std}$$ 

***mean*** - Średnia każdej cechy

### b) Czym jest OneHotEncoder (i kodowanie „one hot” ogólnie)? 

Ponieważ zmienne kategoryczne takie jak np. wielkość lub kolory, nie mogą być bezpośrednio wykorzystane w procesie uczenia maszynowego to muszą zostać one przekształcone na wartość liczbową. OneHotEncoder tworzy oddzielną zmienną typu bool, która określa czy dany rekord należy do podanej kategorii (1 - jeśli tak, 0 - jeśli nie)

### c) Ile neuronów ma warstwa wejsciowa i wyjściowa (czym jest X_train.shape i y_encoded.shape)?

```py
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(y_encoded.shape[1], activation='softmax')
    ])
```

W tym kodzie liczba neuronów wejściowych jest definiowana przez parametr input_shape i **X_train.shape** jest to liczba cech w danym dataset. Natomiast **y_encoded.shape** odpowiada liczbie unikalnych kategorii, które zostały utworzone przez obiekt OneHotEncoder. Liczba tych unikalnych kategorii jest też równa liczbie neuronóœ w warstwie wyjściowej.

### d) Czy funkcja aktywacji relu jest najlepsza do tego zadania?

| funkcja | dokładność testów |
| --- | --- |
| ***Relu*** | 91.11% |
| ***tanh*** | 93.33% |
| ***sigmoid*** | 93.33% |
| ***softmax*** | 55.56% |
| ***linear*** | 93.33% |

Według przeprowadzonych pomiarów funkcja Relu nie jest najlepszą ze wszystkich możliwych funkcji aktywacji. Lepszymi są **tanh**, **sigmoid** i **linear**

### e) Czy różne optymalizatory lub funkcje straty dają różne wyniki?

Dalsze testy będę przeprowadzał korzystając z funkcji **tanh**

| optymalizator | funkcja straty | Prędkość uczenia | dokładność testów |
| --- | --- | --- | --- |
| ***adam*** | *Klasyfikacja wieloklasowa* | 0.001 | 93.33% |
| ***adam*** | *Klasyfikacja wieloklasowa* | 0.01 | 93.33% |
| ***adam*** | *Klasyfikacja wieloklasowa* | 0.1 | 93.33% |
| ***adam*** | *Klasyfikacja binarna* | 0.001 | 93.33% |
| ***adam*** | *Klasyfikacja binarna* | 0.01 | 93.33% |
| ***adam*** | *Klasyfikacja binarna* | 0.1 | 93.33% |
| ***adam*** | *Regresja* | 0.001 | 93.33% |
| ***adam*** | *Regresja* | 0.01 | 93.33% |
| ***adam*** | *Regresja* | 0.1 | 86.67% |
| ***SGD*** | *Klasyfikacja wieloklasowa* | 0.001 | 77.78% |
| ***SGD*** | ***Klasyfikacja wieloklasowa*** | **0.01** | **95.56%** |
| ***SGD*** | *Klasyfikacja wieloklasowa* | 0.1 | 93.33% |
| ***SGD*** | *Klasyfikacja binarna* | 0.001 | 57.78% |
| ***SGD*** | *Klasyfikacja binarna* | 0.01 | 80.00% |
| ***SGD*** | *Klasyfikacja binarna* | 0.1 | 93.33% |
| ***SGD*** | *Regresja* | 0.001 | 55.56% |
| ***SGD*** | *Regresja* | 0.01 | 82.22% |
| ***SGD*** | *Regresja* | 0.1 | 93.33% |


Przykład kodu, dzięki któremu może ustwić prędkość uczenia:

```py
myOptimizer = Adam(learning_rate=1.0)
model.compile(optimizer=myOptimizer, loss='categorical_crossentropy', metrics=['accuracy'])
```

### f) Czy jest sposób, by zmodyfikować tę linię tak, aby rozmiar partii był równy 4 lub 8 lub 16?

Możemy ustalić rozmiar partii, jednak przy wybranych parametrach, wpływa to negatywnie na rezultat. Dokładność modelu spada z **95.56%** na **93.33%**

**Jak wyglądają krzywe uczenia się dla różnych parametrów? Jak zmiana partii wpływa na kształt krzywych?**

### e) Co możesz powiedzieć o wydajności sieci neuronowej na podstawie krzywych uczenia?

Sieć neuronowa uzyskała najlepszą wydajność już w około 16 epoce, jednak potem chwilowo spadła aż do epoki 65.

**Czy ta krzywa sugeruje dobrze dopasowany model, czy mamy do czynienia z niedouczeniem lub przeuczeniem?**

Krzywa oznacza dobrze dopasowany model, ponieważ dokładność modelu dla zbioru treningowego nie jest znacznie różna od zbioru treningowego, a dokładność dla zbioru treningowego nie wzrosła pod koniec i nie jest bardzo bliska 100%.