# Ideen

## Optimierung

### Rechenzeit:
- Erstellen von Label und Imagedata Matrizen

### Preprocessing STFT:

- MEL vs. Linear vs. Log Scale für Frequenz (Mel oder log scheinbar beides gleich gut)
- Spectrogram Bildauflösung erhöhen oder senken
- Spectrogram Format ändern (nicht AxA sondern AxB mit A>B oder B>A)
- Native Samplerate von Audiofile bei Spectrogramerstellung beibehalten
  - bisher wurde Standard-Rate (22050Hz) von Librosa für alle File genutzt
- Frame- und Hopsize bei DFT anpassen, z.B. Hopsize kleiner für höhere Auflösung
- auch Clips unter 1sek verwenden
  - entweder oft duplizieren + dranhängen
  - oder durch Rauschen oder Hintergrundgeräusche wie Wind auffüllen
  - oder einfach Stille einfügen
- Colorcoding Scale anpassen:
  - Amplituden unter einem Threshhold schwarz anzeigen
  - feste Range anstatt dynamischer verwenden
- Interpolation zwischen "Kasten"
- Bilddaten nicht Linear normalisieren (bisher wird "img_arr = img_arr / 255" verwendet, z.B. Softmax benutzen)

### Preprocessing / Feature Extraction Wavelet Transform
- Detailkoeffizienten berechnen --> Features wie Entropie berechnen und speichern (z.B. in CSV)
  - mögliche Feature: Shanon-Entropie, Varianz, Mean of absolute value per subband, standardabweichung pro subband

### Training:
- Epochen erhöhen
- Vortrainiertes Netz von Google verwenden
- Activation Function ändern
- Anzahl an Convolutional Layers
- Anzahl an Deep Layers
- Auflösung von Convolutional oder Deep Layers ändern
- Change Learning Rate
  - change rate over time (https://www.tensorflow.org/tutorials/keras/overfit_and_underfit)

### Evaluation:

- Crossvalidation
  - Accuracy pro Epoche ermitteln und ausgeben:
    - Bsp. 10-fold: 8 train folds, 1 fold für Zwischenergebnis, 1 fold für Finale Validation 
