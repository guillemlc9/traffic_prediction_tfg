"""
create_sequences.py
-------------------
Crea seqüències temporals (sliding windows) per entrenar models LSTM.
Genera conjunts X_train, y_train, X_val, y_val, X_test, y_test.
"""

import sys
from pathlib import Path

# Afegim el directori arrel al PYTHONPATH
root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root_dir))

import polars as pl
import numpy as np
from typing import Dict, Tuple
from src.lstm.prepare_data import get_all_trams_data
from src.lstm.normalize_data import normalize_traffic_data


def create_all_sequences(
    df: pl.DataFrame,
    window_size: int = 36,
    horizon: int = 1,
    value_col: str = "estatActual_norm"
) -> Dict[str, np.ndarray]:
    """
    Crea seqüències globals sense barrejar sèries entre trams.
    Les seqüències es classifiquen en train/val/test segons l'split del target.
    """
    # Ens assegurem de l'ordre correcte
    df = df.sort(["idTram", "timestamp"])
    pdf = df.to_pandas()

    X_train, y_train = [], []
    X_val,   y_val   = [], []
    X_test,  y_test  = [], []

    for tram_id, g in pdf.groupby("idTram"):
        v = g[value_col].astype("float32").to_numpy()
        s = g["split"].to_numpy()

        n = len(g)
        max_start = n - window_size - horizon + 1
        if max_start <= 0:
            print(f"Tram {tram_id} amb massa pocs punts ({n}), s'ignora.")
            continue

        for start in range(max_start):
            end = start + window_size
            target_idx = end + horizon - 1

            window = v[start:end]
            target = v[target_idx]
            split_target = s[target_idx]

            window = window.reshape(window_size, 1)

            if split_target == "train":
                X_train.append(window)
                y_train.append(target)
            elif split_target == "val":
                X_val.append(window)
                y_val.append(target)
            elif split_target == "test":
                X_test.append(window)
                y_test.append(target)
            else:
                raise ValueError(f"Split desconegut: {split_target}")

    def to_array(xs, ys):
        if len(xs) == 0:
            return np.empty((0, window_size, 1), dtype="float32"), np.empty((0,), dtype="float32")
        return np.stack(xs).astype("float32"), np.array(ys, dtype="float32")

    X_train, y_train = to_array(X_train, y_train)
    X_val,   y_val   = to_array(X_val,   y_val)
    X_test,  y_test  = to_array(X_test,  y_test)

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val":   X_val,
        "y_val":   y_val,
        "X_test":  X_test,
        "y_test":  y_test,
    }


def print_sequences_summary(sequences: Dict[str, np.ndarray]):
    """
    Mostra un resum de les seqüències generades.
    """
    
    for split_name in ['train', 'val', 'test']:
        X_key = f'X_{split_name}'
        y_key = f'y_{split_name}'
        
        if X_key in sequences:
            X = sequences[X_key]
            y = sequences[y_key]
            
            print(f"\n{split_name.upper()}:")
            print(f"  X_{split_name}: {X.shape} → (n_samples, window_size, n_features)")
            print(f"  y_{split_name}: {y.shape} → (n_samples,)")
            print(f"  Rang X: [{X.min():.3f}, {X.max():.3f}]")
            print(f"  Rang y: [{y.min():.3f}, {y.max():.3f}]")
            print(f"  Memòria X: {X.nbytes / 1024 / 1024:.2f} MB")
            print(f"  Memòria y: {y.nbytes / 1024 / 1024:.2f} MB")


def save_sequences(
    sequences: Dict[str, np.ndarray],
    output_dir: str = "data/processed/lstm_sequences"
):
    """
    Guarda les seqüències en format .npy per carregar-les ràpidament.
    
    Args:
        sequences: Diccionari amb X_train, y_train, etc.
        output_dir: Directori on guardar els fitxers
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for key, array in sequences.items():
        filepath = output_path / f"{key}.npy"
        np.save(filepath, array)



def load_sequences(
    input_dir: str = "data/processed/lstm_sequences"
) -> Dict[str, np.ndarray]:
    """
    Carrega les seqüències des de fitxers .npy.
    
    Args:
        input_dir: Directori on es troben els fitxers
    
    Returns:
        Diccionari amb X_train, y_train, etc.
    """
    input_path = Path(input_dir)
    sequences = {}
    
    for split_name in ['train', 'val', 'test']:
        for prefix in ['X', 'y']:
            key = f"{prefix}_{split_name}"
            filepath = input_path / f"{key}.npy"
            
            if filepath.exists():
                sequences[key] = np.load(filepath)
    
    return sequences


def main():
    """
    Script de preparació de seqüències per al model LSTM global.
    - Carrega el dataset imputat
    - Assigna splits temporals (train/val/test)
    - Normalitza estatActual per tram (usant només train)
    - Construeix seqüències globals (tots els trams)
    - Guarda X_train, y_train, X_val, y_val, X_test, y_test en .npy
    """
    # Configuració
    WINDOW_SIZE = 36  # 3 hores (36 * 5 minuts)
    HORIZON = 1       # Predir el següent timestep (5 minuts)
    DATA_PATH = "data/processed/dataset_imputed_clean.parquet"
    OUTPUT_DIR = "data/processed/lstm_sequences"

    print(f"Window size: {WINDOW_SIZE} timesteps (3 hores)")
    print(f"Horizon: {HORIZON} timestep (5 minuts)")
    print(f"Input data: {DATA_PATH}")

    # 1. Carreguem dades
    df = pl.read_parquet(DATA_PATH)
    print(f"   Shape original: {df.shape}")

    # 2. Preparem dades amb splits
    info = get_all_trams_data(df)
    df_with_splits = info["df"]
    print(f"   Shape després de filtrar trams seleccionats: {df_with_splits.shape}")

    # 3. Normalitzem dades
    df_normalized, normalizer = normalize_traffic_data(df_with_splits)
    print(f"   Columnes disponibles: {df_normalized.columns}")

    # 4. Creem seqüències globals
    sequences = create_all_sequences(
        df_normalized,
        window_size=WINDOW_SIZE,
        horizon=HORIZON,
        value_col="estatActual_norm",
    )

    # 5. Mostram resum
    print_sequences_summary(sequences)

    # 6. Guardem seqüències
    save_sequences(sequences, output_dir=OUTPUT_DIR)

    print("Processat completat!")
    print(f"Seqüències globals guardades a: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
