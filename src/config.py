"""Configuración central del proyecto MotoTaller IA (análisis de opiniones)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Dataset público de Hugging Face (opiniones en español, etiqueta binaria 0/1).
HF_DATASET = "sepidmnorozy/Spanish_sentiment"
# Alternativa: un CSV propio con columnas `text` y `label` (0 = negativo, 1 = positivo)
DATA_CSV = ROOT / "data" / "opiniones.csv"

# ¡Verifica con la celda de ejemplos del notebook! La ficha del dataset no define
# el significado de 0/1; por los ejemplos se interpreta 0 = negativo, 1 = positivo.
LABEL_NAMES = {0: "Negativo", 1: "Positivo"}

MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
MODEL_PATH = MODELS_DIR / "opiniones_rnn.keras"
INFO_PATH = MODELS_DIR / "model_info.json"

MAX_TOKENS = 10000   # tamaño del vocabulario
MAX_LEN = 100        # longitud fija de cada secuencia (en palabras)
EMBED_DIM = 64
BATCH_SIZE = 32
EPOCHS = 30
SEED = 42

# Modelos a comparar; se guarda el de mejor F1 macro de validación (con su umbral óptimo)
MODEL_TYPES = ("gru", "lstm")

# Split 70 / 15 / 15
VAL_SIZE = 0.15
TEST_SIZE = 0.15
