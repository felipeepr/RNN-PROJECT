"""Predicción con el modelo entrenado.

Uso:  python -m src.predict "El servicio fue excelente, muy recomendado"
"""
import json
import sys

import numpy as np
import tensorflow as tf

from . import config
from .data import clean_text

_model = None


def load_model():
    """Carga (una sola vez) el modelo guardado."""
    global _model
    if _model is None:
        from tensorflow import keras

        _model = keras.models.load_model(config.MODEL_PATH)
    return _model


def load_info() -> dict:
    return json.loads(config.INFO_PATH.read_text(encoding="utf-8")) if config.INFO_PATH.exists() else {}


def predict_texts(texts):
    """Devuelve una lista de (etiqueta, confianza, prob_positivo) por texto."""
    model = load_model()
    cleaned = [clean_text(t) for t in texts]
    p = model.predict(tf.constant(cleaned), verbose=0).reshape(-1)
    thr = float(load_info().get("umbral", 0.5))  # umbral elegido en validación
    out = []
    for prob in p:
        label = int(prob >= thr)
        conf = float(prob if label == 1 else 1 - prob)
        out.append((config.LABEL_NAMES[label], conf, float(prob)))
    return out


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit('Uso: python -m src.predict "texto de la opinión"')
    label, conf, _ = predict_texts([" ".join(sys.argv[1:])])[0]
    print(f"{label} ({conf:.1%})")
