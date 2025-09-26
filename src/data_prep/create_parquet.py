#!/usr/bin/env python3
"""
create_dataset_1y.py
--------------------
Uneix tots els fitxers CSV de trànsit (1 any) i genera dataset_1y.parquet
amb timestamps en format datetime.

Executa des de l'arrel del repositori:
    python src/create_dataset_1y.py
"""

import polars as pl
from pathlib import Path

# --- Configuració -----------------------------------------------------------

# Carpeta amb els CSV d'origen (canvia si la teva estructura és diferent)
RAW_DIR = Path("/Users/guillemlopezcolomer/Library/CloudStorage/GoogleDrive-guillemlc9@gmail.com/My Drive/TFG/Data/Open Data BCN/Trànsit/dades")        # ex: traffic-prediction-tfg/data/raw
OUTPUT  = Path("/Users/guillemlopezcolomer/Desktop/traffic_prediction_tfg/data")  # on guardarem el parquet
OUTPUT.mkdir(parents=True, exist_ok=True)

PARQUET_FILE = OUTPUT / "dataset_1y.parquet"

# ---------------------------------------------------------------------------

def read_and_clean_csv(csv_path: Path) -> pl.DataFrame:
    """Llegeix un CSV i fa la conversió del camp 'data' a timestamp real."""
    df = pl.read_csv(
        csv_path,
        schema_overrides={
            "idTram": pl.Int64,
            "data": pl.Utf8,
            "estatActual": pl.Int64,
            "estatPrevist": pl.Int64,
        }
    )

    # Converteix 'data' (YYYYMMDDHHMMSS) a datetime i crea columna 'timestamp'
    df = df.with_columns(
        pl.col("data")
          .str.strptime(pl.Datetime, format="%Y%m%d%H%M%S")
          .alias("timestamp")
    )
    return df

def main():
    csv_files = sorted(RAW_DIR.glob("*.csv"))[64:76]
    if not csv_files:
        raise FileNotFoundError(f"No s'han trobat CSV a {RAW_DIR}")

    print(f"Llegint {len(csv_files)} fitxers CSV…")
    dfs = [read_and_clean_csv(f) for f in csv_files]

    print("Concatenant…")
    df_all = pl.concat(dfs, how="vertical")

    # Ordena per tram i temps, elimina duplicats exactes
    df_all = (
        df_all.unique()
              .sort(["idTram", "timestamp"])
    )

    print(f"Guardant Parquet: {PARQUET_FILE}")
    df_all.write_parquet(PARQUET_FILE, compression="zstd")
    print("Fet ✅")

if __name__ == "__main__":
    main()
