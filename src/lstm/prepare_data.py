"""
prepare_lstm.py
----------------
Preparem les dades per entrenar un model LSTM per a cada tram.
"""

import sys
from pathlib import Path

# Afegir el directori arrel al PYTHONPATH
root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root_dir))

import polars as pl
from src.data_prep.prepare_time_splits import get_temporal_splits, prepare_tram_series, SELECTED_TRAMS
from typing import Dict, List
from datetime import datetime

def add_split_col(df: pl.DataFrame, splits: Dict[str, datetime]) -> pl.DataFrame:
    """
    Afegeix una columna 'split' (train/val/test) en funció del timestamp.
    """
    return (
        df
        .with_columns(
            pl.when(pl.col("timestamp") < splits["train_end"])
              .then(pl.lit("train"))
            .when(pl.col("timestamp") < splits["val_end"])
              .then(pl.lit("val"))
            .otherwise(pl.lit("test"))
            .alias("split")
        )
        .select(["idTram", "timestamp", "estatActual", "split"])
        .sort(["idTram", "timestamp"])
    )

def get_all_trams_data(
    df: pl.DataFrame, 
    tram_ids: List[int] = None
) -> Dict[str, any]:
    if tram_ids is None:
        tram_ids = SELECTED_TRAMS
    
    # Splits temporals globals
    splits_info = get_temporal_splits(df)
    
    # Filtrar dataset pels trams seleccionats
    df_filtered = df.filter(pl.col('idTram').is_in(tram_ids))
    
    # Afegir columna 'split'
    df_with_split = add_split_col(df_filtered, splits_info)
    
    return {
        'splits': splits_info,
        'tram_ids': tram_ids,
        'df': df_with_split,  # <-- df llarg amb idTram, timestamp, estatActual, split
        'get_tram_series': lambda tram_id, split: prepare_tram_series(
            df_filtered, tram_id, split, splits_info
        )
    }

def main():
    """
    Ús del mòdul.
    """
    # Carregar dades
    DATA_PATH = "data/processed/dataset_imputed_clean.parquet"
    print(f"Carregant dades de {DATA_PATH}...")
    df = pl.read_parquet(DATA_PATH)
    
    info = get_all_trams_data(df)
    df_all = info["df"]
    
    print(df_all.tail())


if __name__ == "__main__":
    main()