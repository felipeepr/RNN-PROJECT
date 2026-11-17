"""Modelos recurrentes (GRU / LSTM bidireccionales) con vectorización integrada."""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from . import config


def build_vectorizer(train_texts):
    """TextVectorization ajustada SOLO con el conjunto de entrenamiento."""
    vec = layers.TextVectorization(
        max_tokens=config.MAX_TOKENS,
        output_mode="int",
        output_sequence_length=config.MAX_LEN,
        name="vectorizer",
    )
    vec.adapt(tf.constant(list(train_texts)))
    return vec


def build_model(kind: str, vectorizer) -> keras.Model:
    """kind: 'gru' o 'lstm'. La entrada es texto crudo (string)."""
    rnn = {"gru": layers.GRU, "lstm": layers.LSTM}[kind]
    inputs = keras.Input(shape=(), dtype=tf.string, name="texto")
    x = vectorizer(inputs)
    x = layers.Embedding(config.MAX_TOKENS, config.EMBED_DIM, mask_zero=True)(x)
    x = layers.Bidirectional(rnn(64, dropout=0.2))(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs, name=f"opiniones_{kind}")
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model
