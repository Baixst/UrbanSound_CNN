# Ideen

## Optimierung

### Rechenzeit:
- Erstellen von Label und Imagedata Matrizen

### Preprocessing:

- MEL Scale für Frequenz
- Spectrogram Bildauflösung erhöhen
- Spectrogram Format ändern (nicht AxA sondern AxB mit A>B oder B>A)
- Frame- und Hopsize bei DFT anpassen, z.B. Hopsize kleiner für höhere Auflösung
- auch Clips unter 1sek verwenden
  - entweder oft duplizieren + dranhängen
  - oder durch Rauschen oder Hintergrundgeräusche wie Wind auffüllen
- Colorcoding Scale anpassen:
  - Amplituden unter einem Threshhold schwarz anzeigen
  - feste Range anstatt dynamischer verwenden

### Training:
- Epochen erhöhen
- Vortrainiertes Netz von Google verwenden
- Aktivation Function ändern
- Anzahl an Convolutional Layers
- Anzahl an Deep Layers
- Auflösung von Convolutional oder Deep Layers ändern

### Evaluation:

- Testaccuracy pro Klasse ermitteln
- Crossvalidation
