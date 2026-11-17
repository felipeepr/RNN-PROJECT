# 🏍️ MotoTaller IA: análisis de opiniones con redes recurrentes

Proyecto de Deep Learning con **datos secuenciales (texto)**: una red recurrente bidireccional (**GRU o LSTM**, TensorFlow/Keras) lee la opinión de un cliente y la clasifica como **positiva o negativa**. Una app en Streamlit permite escribir una opinión nueva y ver el sentimiento con su nivel de confianza. Caso de uso: un taller de motos que quiere detectar rápido las opiniones negativas.

## Dataset

**`sepidmnorozy/Spanish_sentiment`** (Hugging Face): [huggingface.co/datasets/sepidmnorozy/Spanish_sentiment](https://huggingface.co/datasets/sepidmnorozy/Spanish_sentiment). Opiniones en español (reseñas de hoteles) con columnas `text` y `label` (0/1). Se descarga automáticamente en el notebook con la librería `datasets`.

- Se unen los tres splits originales (1.472 filas), se limpian y se vuelven a dividir de forma propia. Tras la limpieza quedan **1.370 opiniones: 239 negativas y 1.131 positivas**, un dataset **desbalanceado** (~83 % positivas).
- La ficha pública no indica licencia ni fuente. Por los ejemplos impresos en el notebook se confirma **0 = negativo y 1 = positivo**.
- **Limitación:** son opiniones de hoteles, no de talleres de moto. El modelo aprende sentimiento general en español, que se transfiere solo parcialmente al dominio de talleres. Es una prueba de concepto.

## Estructura

```
├── notebooks/experimentacion_rnn.ipynb   # Experimentación completa ejecutada en Google Colab
├── src/
│   ├── config.py     # Rutas e hiperparámetros
│   ├── data.py       # Carga, limpieza y split 70/15/15
│   ├── model.py      # TextVectorization + Embedding + Bi-GRU/Bi-LSTM
│   ├── train.py      # Entrena GRU y LSTM, evalúa, grafica y guarda el mejor
│   └── predict.py    # Carga del modelo y predicción (también por consola)
├── app.py            # Aplicación Streamlit
├── models/           # opiniones_rnn.keras y model_info.json (modelo entrenado)
├── reports/          # Salidas de src/train.py (curvas, matriz, predicciones)
└── requirements.txt
```

## Metodología

1. **Exploración:** balance de clases, nulos y duplicados, longitud de las opiniones (mediana 12 palabras, p95 44) y palabras más frecuentes por clase.
2. **Limpieza:** se eliminan nulos, URLs, espacios sobrantes, opiniones de menos de 3 palabras y **duplicados** (102 filas descartadas). Se conservan las tildes.
3. **División:** estratificada 70 % train (958) / 15 % validación (206) / 15 % test (206).
4. **Vectorización:** capa `TextVectorization` (vocabulario de 10.000 palabras, secuencias de 100) **dentro del modelo**, ajustada solo con train. La app envía texto crudo y no hay diferencias entre entrenamiento y predicción.
5. **Modelo:** `Embedding(64)` → `Bidirectional(GRU o LSTM, 64)` → `Dense(32)` → `Dropout(0.4)` → `Dense(1, sigmoide)`. Se entrenan **GRU y LSTM** y se guarda el de mejor F1 macro en validación.
6. **Desbalance de clases:** **pesos de clase** al entrenar y **umbral de decisión** elegido en validación para maximizar el F1 macro (resultó 0.27 en lugar de 0.5).
7. **Evaluación:** accuracy frente a la línea base de la clase mayoritaria, balanced accuracy, F1 macro, precisión/recall/F1 por clase, ROC-AUC, matriz de confusión, curva ROC y 12 predicciones sobre el conjunto de test.

## Resultados

- Modelos comparados (F1 macro en validación): GRU = 0.779, LSTM = 0.774. Elegido: **GRU bidireccional**, umbral 0.27.
- **Accuracy en test: 0.864**, frente a una línea base de 0.825 (siempre responder "positivo").
- Balanced accuracy: 0.677 | F1 macro: 0.711 | ROC-AUC: 0.866.

| Clase | Precisión | Recall | F1 | Soporte |
|---|---|---|---|---|
| Negativo | 0.70 | 0.39 | 0.50 | 36 |
| Positivo | 0.88 | 0.96 | 0.92 | 170 |

### Análisis

- El accuracy (86,4 %) supera solo ligeramente la línea base (82,5 %): **no es la métrica que cuenta la historia**. El F1 macro (0,71) y el recall por clase muestran mejor el desempeño.
- El modelo detecta bien las opiniones positivas (recall 0,96) pero **solo encuentra 14 de las 36 negativas** (recall 0,39): 22 negativas se leyeron como positivas. Es el error más relevante para un taller.
- La GRU muestra **cierto sobreajuste** (train 0,99 frente a val 0,87 en la mejor época), esperable con solo 958 opiniones de entrenamiento.
- Causas probables: pocas opiniones negativas (239), etiquetas con algo de ruido (hay opiniones neutras marcadas como positivas) y dominio distinto al de un taller.
- Mejoras posibles: más opiniones negativas (reales de talleres), embeddings preentrenados en español o un modelo tipo transformer, más regularización y una clase neutral.

## Ejecución

Requiere **Python 3.10 a 3.12** (TensorFlow aún no publica paquetes para versiones más nuevas).

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Aplicación Streamlit** (usa el modelo ya entrenado en `models/`):

```bash
streamlit run app.py
```

Escribe una opinión (o usa los ejemplos) y pulsa **Analizar**: la app muestra el sentimiento, la confianza y las probabilidades de cada clase.

**Predicción por consola:**

```bash
python -m src.predict "Pésimo servicio, no vuelvo"
```

**Reproducir el entrenamiento:**
- *Google Colab:* abre `notebooks/experimentacion_rnn.ipynb` y ejecuta todas las celdas (descarga el dataset solo). Al final se descargan `opiniones_rnn.keras` y `model_info.json` para guardarlos en `models/`.
- *Local:* `python -m src.train` (descarga el dataset, entrena, y deja el modelo en `models/` y los gráficos en `reports/`).

## Limitaciones

- Entrenado con opiniones de hoteles; en opiniones de talleres de moto puede equivocarse más (por ejemplo, algunas opiniones negativas claras sobre un taller se clasifican como positivas).
- Pocas opiniones negativas: detectarlas es la parte difícil del problema (recall de `Negativo` = 0,39).
- Solo distingue positivo y negativo (no hay clase neutral) y no entiende ironía ni sarcasmo.
- Es una ayuda de apoyo para priorizar opiniones, no una evaluación definitiva.
