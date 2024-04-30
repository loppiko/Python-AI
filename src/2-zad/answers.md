### a) Co się dzieje w preprocessing? Do czego służy funkcja reshape, to_categorical i np.argmax?

W preprocessing przetwarzamy obraz do takiego stanu, w którym łatwiej będzie nauczyć naszą sieć.


**reshape** - zmienia kształt każdego obrazu na podany format.

```py
train_images = train_images.reshape((train_images.shape[0], 28, 28, 1)).astype('float32') / 255
```

W tym przypdaku będzie to obraz 28x28px


**to_categorical** - przkształca etykietu dataset na zero jedynkowe zmienne

**arg_max** - zamienia sporwrotem etykiety prezkształcone funkcją *to_categorizal* na ich pierwotne odpowiedniki


### b) Jak dane przepływają przez sieć i jak się w niej transformują? Co każda z warstw dostaje na wejście i co wyrzuca na wyjściu?

- **Conv2D**
  - Wejście: czarnobiałe obrazy 28x28px
  - Wyjście: obrazki przedstawiwające dane cechy, przetworzone przez filtry (32)
- **MaxPoolinx2D**
  - Wejście: obrazki przedstawiwające dane cechy
  - Wyjście: obrazki o zredukowanej liczbie cech
- **Flatten**
  - Wejście: obrazki o zredukowanej liczbie cech
  - Wyjście: spłaszczony wektor
- **Dense(64, relu)**
  - Wejście: spłaszczony wektor
  - Wyjście: wektor o długości 64 z aktywacją relu.
- **Dense(10, softmax)**
  - Wejście: wektor o długości 64 z aktywacją relu.
  - Wyjście: wektor reprezentujący prawdopodobieństwa przynależności do 10 różnych klas

### c) Jakich błędów na macierzy błędów jest najwięcej. Które cyfry są często mylone z jakimi innymi?

Największą szansa na błąd występuje gdy wejsciem jest 3 i liczba zotsanie pomylona z 9

### d) Co możesz powiedzieć o krzywych uczenia się. Czy mamy przypadek przeuczenia lub niedouczenia się?

Tempo uczenia się rośnie szczególnie do momentu 3 epoki, potem spowalnia. Mamy doczynienie z delikatnym niedouczeniem.

### e) Jak zmodyfikować kod programu, aby model sieci był zapisywany do pliku h5 co epokę, pod warunkiem, że w tej epoce osiągnęliśmy lepszy wynik?

```py
checkpoint_callback = ModelCheckpoint(filepath='best_model.h5', monitor='val_accuracy', mode='max', save_best_only=True)

model.fit(train_images, train_labels, epochs=5, batch_size=64, validation_split=0.2, callbacks=[checkpoint_callback, history])
```

monitoruje wartość dokładności (accuracy) na zbiorze walidacyjnym i w momencie znalezienia lepszego nadpisuje obecny model.