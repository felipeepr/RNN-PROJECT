"""Entrenamiento, comparación GRU vs LSTM, evaluación y guardado.

El dataset está desbalanceado (pocas opiniones negativas), por eso:
  - se usan pesos de clase al entrenar,
  - el umbral de decisión se elige en VALIDACIÓN para maximizar el F1 macro,
  - se reportan balanced accuracy, F1 macro y la línea base de la clase mayoritaria.

Uso:  python -m src.train
"""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score, roc_auc_score)
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras

from . import config
from .data import load_raw, prepare, split_data
from .model import build_model, build_vectorizer


def best_threshold(y, p):
    """Umbral que maximiza el F1 macro (se calcula con validación)."""
    ths = np.arange(0.05, 0.96, 0.01)
    f1s = [f1_score(y, (p >= t).astype(int), average="macro") for t in ths]
    i = int(np.argmax(f1s))
    return float(ths[i]), float(f1s[i])


def plot_history(histories, out):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for kind, h in histories.items():
        ax[0].plot(h["accuracy"], label=f"{kind} train")
        ax[0].plot(h["val_accuracy"], "--", label=f"{kind} val")
        ax[1].plot(h["loss"], label=f"{kind} train")
        ax[1].plot(h["val_loss"], "--", label=f"{kind} val")
    ax[0].set(title="Accuracy", xlabel="Época")
    ax[1].set(title="Loss", xlabel="Época")
    ax[0].legend()
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def plot_confusion(y_true, y_pred, out):
    names = [config.LABEL_NAMES[i] for i in (0, 1)]
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=names, yticklabels=names, ax=ax)
    ax.set(xlabel="Predicción", ylabel="Real", title="Matriz de confusión (test)")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def main():
    tf.keras.utils.set_random_seed(config.SEED)
    config.MODELS_DIR.mkdir(exist_ok=True)
    config.REPORTS_DIR.mkdir(exist_ok=True)

    df = prepare(load_raw())
    train, val, test = split_data(df)
    print(f"total={len(df)} train={len(train)} val={len(val)} test={len(test)}")
    print(df["label"].value_counts().to_string())

    vectorizer = build_vectorizer(train["text"])
    tx, ty = tf.constant(train["text"].tolist()), train["label"].values
    vx, vy = tf.constant(val["text"].tolist()), val["label"].values

    cw_arr = compute_class_weight("balanced", classes=np.array([0, 1]), y=ty)
    class_weight = {0: float(cw_arr[0]), 1: float(cw_arr[1])}
    print("Pesos de clase:", class_weight)

    models, histories, val_f1, thrs = {}, {}, {}, {}
    for kind in config.MODEL_TYPES:
        print(f"\n=== Entrenando {kind.upper()} ===")
        model = build_model(kind, vectorizer)
        cbs = [
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2),
        ]
        h = model.fit(tx, ty, validation_data=(vx, vy), epochs=config.EPOCHS,
                      batch_size=config.BATCH_SIZE, callbacks=cbs, class_weight=class_weight)
        models[kind], histories[kind] = model, h.history
        pv = model.predict(vx, verbose=0).reshape(-1)
        thrs[kind], val_f1[kind] = best_threshold(vy, pv)

    best = max(val_f1, key=val_f1.get)
    model, thr = models[best], thrs[best]
    print("\nF1 macro en validación:", {k: round(v, 4) for k, v in val_f1.items()},
          "-> se guarda:", best.upper(), f"con umbral {thr:.2f}")

    sx, sy = tf.constant(test["text"].tolist()), test["label"].values
    prob = model.predict(sx, verbose=0).reshape(-1)
    pred = (prob >= thr).astype(int)
    acc = float((pred == sy).mean())
    baseline = float(max((sy == 0).mean(), (sy == 1).mean()))
    bacc = float(balanced_accuracy_score(sy, pred))
    f1m = float(f1_score(sy, pred, average="macro"))
    auc = float(roc_auc_score(sy, prob))
    report = classification_report(sy, pred, target_names=[config.LABEL_NAMES[0], config.LABEL_NAMES[1]])
    print(f"Umbral: {thr:.2f} | Test accuracy: {acc:.4f} (línea base clase mayoritaria: {baseline:.4f})")
    print(f"Balanced accuracy: {bacc:.4f} | F1 macro: {f1m:.4f} | ROC-AUC: {auc:.4f}\n{report}")
    (config.REPORTS_DIR / "classification_report.txt").write_text(report)

    plot_history(histories, config.REPORTS_DIR / "training_curves.png")
    plot_confusion(sy, pred, config.REPORTS_DIR / "confusion_matrix.png")

    n = min(12, len(test))
    idx = np.random.default_rng(config.SEED).choice(len(test), n, replace=False)
    table = pd.DataFrame({
        "texto": test["text"].iloc[idx].str.slice(0, 120).values,
        "real": [config.LABEL_NAMES[i] for i in sy[idx]],
        "prediccion": [config.LABEL_NAMES[i] for i in pred[idx]],
        "prob_positivo": prob[idx].round(3),
    })
    table["acierto"] = table["real"] == table["prediccion"]
    table.to_csv(config.REPORTS_DIR / "predicciones_test.csv", index=False)
    print(table.to_string())

    model.save(config.MODEL_PATH)
    config.INFO_PATH.write_text(json.dumps({
        "tipo": best, "umbral": thr, "test_accuracy": acc, "linea_base": baseline,
        "balanced_accuracy": bacc, "f1_macro": f1m, "roc_auc": auc, "val_f1_macro": val_f1,
    }, indent=2), encoding="utf-8")
    print(f"Modelo guardado en {config.MODEL_PATH}")


if __name__ == "__main__":
    main()
