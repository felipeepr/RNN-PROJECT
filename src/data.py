"""Carga, limpieza y división del dataset de opiniones."""
import re

import pandas as pd
from sklearn.model_selection import train_test_split

from . import config

_URL = re.compile(r"https?://\S+|www\.\S+")
_SPACES = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Limpieza ligera: quita URLs y espacios sobrantes.

    Se conservan tildes y signos útiles; la minúscula y la puntuación las maneja
    la capa TextVectorization dentro del modelo. La app usa esta MISMA función.
    """
    text = _URL.sub(" ", str(text))
    return _SPACES.sub(" ", text).strip()


def load_raw() -> pd.DataFrame:
    """Devuelve un DataFrame con columnas `text` y `label`."""
    if config.DATA_CSV.exists():
        df = pd.read_csv(config.DATA_CSV)
    else:
        from datasets import load_dataset

        ds = load_dataset(config.HF_DATASET)
        df = pd.concat([ds[s].to_pandas() for s in ds.keys()], ignore_index=True)
    return df[["text", "label"]]


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Limpieza: texto nulo/vacío, duplicados y opiniones demasiado cortas."""
    df = df.dropna(subset=["text", "label"]).copy()
    df["text"] = df["text"].map(clean_text)
    df["label"] = df["label"].astype(int)
    df = df[df["text"].str.split().str.len() >= 3]
    df = df.drop_duplicates(subset="text").reset_index(drop=True)
    return df


def split_data(df: pd.DataFrame):
    """Split estratificado train/val/test (70/15/15)."""
    trval, test = train_test_split(
        df, test_size=config.TEST_SIZE, stratify=df["label"], random_state=config.SEED
    )
    val_rel = config.VAL_SIZE / (1 - config.TEST_SIZE)
    train, val = train_test_split(
        trval, test_size=val_rel, stratify=trval["label"], random_state=config.SEED
    )
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)
