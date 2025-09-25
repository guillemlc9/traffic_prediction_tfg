"""
gaps.py
-------
Funcions per detectar i analitzar buits temporals en el dataset
de trànsit de Barcelona (registres cada 5 minuts).

Ús bàsic:
---------
from traffic_prediction.data_prep.gaps import find_gaps, classify_gaps, find_global_gaps

df = pl.read_parquet("dataset_1y.parquet")
gaps = find_gaps(df)
gaps = classify_gaps(gaps)
gaps_global = find_global_gaps(df)
"""

import polars as pl

def find_gaps(df: pl.DataFrame,
              id_col: str = "idTram",
              ts_col: str = "timestamp",
              freq_minutes: int = 5,
              tolerance_minutes: int = 1) -> pl.DataFrame:
    """
    Detecta buits temporals per cada tram.

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame amb almenys [id_col, ts_col]
    id_col : str
        Nom de la columna del tram
    ts_col : str
        Nom de la columna de timestamp (Datetime)
    freq_minutes : int
        Freqüència esperada entre lectures
    tolerance_minutes : int
        Marge addicional per considerar un buit (p. ex. 1 min)

    Returns
    -------
    pl.DataFrame
        Columnes: [id_col, gap_start, gap_end, gap_minutes]
    """
    exp = pl.duration(minutes=freq_minutes + tolerance_minutes)

    df = (
        df.sort([id_col, ts_col])
          .with_columns(
              (pl.col(ts_col) - pl.col(ts_col).shift(1))
              .over(id_col)
              .alias("delta")
          )
    )

    gaps = (
        df.filter(pl.col("delta") > exp)
          .with_columns([
              pl.col(ts_col).alias("gap_end"),
              pl.col(ts_col).shift(1).over(id_col).alias("gap_start"),
              pl.col("delta").dt.total_minutes().alias("gap_minutes")
          ])
          .drop_nulls(subset=["gap_start"])
          .select([id_col, "gap_start", "gap_end", "gap_minutes"])
    )
    return gaps


def classify_gaps(gaps: pl.DataFrame) -> pl.DataFrame:
    """
    Afegeix una columna 'gap_type' segons la durada del buit:
      - short  : ≤15 min
      - medium : 20 min – 1 h
      - long   : >1 h
    """
    return gaps.with_columns(
        pl.when(pl.col("gap_minutes") <= 15)
          .then("short")
          .when(pl.col("gap_minutes") <= 60)
          .then("medium")
          .otherwise("long")
          .alias("gap_type")
    )


def find_global_gaps(df: pl.DataFrame,
                     ts_col: str = "timestamp",
                     freq_minutes: int = 5,
                     tolerance_minutes: int = 1) -> pl.DataFrame:
    """
    Detecta buits de la sèrie temporal global
    (sense repetir per cada tram).

    Returns
    -------
    pl.DataFrame amb [gap_start, gap_end, gap_minutes]
    """
    exp = pl.duration(minutes=freq_minutes + tolerance_minutes)

    ts = (
        df.select(ts_col)
          .unique()
          .sort(ts_col)
          .with_columns(
              (pl.col(ts_col) - pl.col(ts_col).shift(1)).alias("delta")
          )
    )

    global_gaps = (
        ts.filter(pl.col("delta") > exp)
          .with_columns([
              pl.col(ts_col).alias("gap_end"),
              pl.col(ts_col).shift(1).alias("gap_start"),
              pl.col("delta").dt.total_minutes().alias("gap_minutes")
          ])
          .drop_nulls(subset=["gap_start"])
          .select(["gap_start", "gap_end", "gap_minutes"])
    )
    return global_gaps
