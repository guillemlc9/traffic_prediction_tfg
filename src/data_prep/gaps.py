"""
gaps.py
-------
Funcions per detectar i analitzar buits temporals en el dataset
de trànsit.

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
              tolerance_minutes: int = 4) -> pl.DataFrame:
    """
    Detecta buits temporals per tram.

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame amb almenys [id_col, ts_col]
    id_col : str
        Nom de la columna del tram
    ts_col : str
        Nom de la columna de timestamp (Datetime)
    freq_minutes : int
        Freqüència esperada entre lectures (en minuts)
    tolerance_minutes : int
        Marge addicional per considerar un buit (en minuts)

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
          .then(pl.lit("short"))
          .when(pl.col("gap_minutes") <= 60)
          .then(pl.lit("medium"))
          .otherwise(pl.lit("long"))
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


def classify_gaps_from_calendar(df: pl.DataFrame, freq_minutes: int = 5) -> pl.DataFrame:
    """
    df: calendari d'UN tram, amb 'timestamp' (ordenable) i 'is_gap' (bool).
    Retorna df + ['gap_minutes', 'gap_type'] per fila.
    """
    if df.is_empty():
        return df.with_columns([
            pl.lit(None, dtype=pl.Int32).alias("gap_minutes"),
            pl.lit(None, dtype=pl.Utf8).alias("gap_type"),
        ])

    base = (
        df.sort("timestamp")
          .with_columns([
              # marca canvi respecte la fila anterior (True al primer)
              (pl.col("is_gap") != pl.col("is_gap").shift(1))
                .fill_null(True)
                .cast(pl.Int32)
                .alias("_chg"),
          ])
          .with_columns([
              # id de run consecutiu
              pl.col("_chg").cum_sum().alias("_run_id"),
              pl.col("is_gap").cast(pl.Int8).alias("_gap"),
          ])
    )

    runs = (
        base.group_by("_run_id")
            .agg([
                pl.col("_gap").first().alias("_run_is_gap"),  # 1 si és gap, 0 si no
                pl.len().alias("_run_len"),
            ])
    )

    freq = pl.lit(freq_minutes, dtype=pl.Int32)
    out = (
        base.join(runs, on="_run_id", how="left")
            .with_columns([
                pl.when(pl.col("_run_is_gap") == 1)
                  .then(pl.col("_run_len") * freq)
                  .otherwise(None)
                  .alias("gap_minutes"),
            ])
            .with_columns([
                pl.when((pl.col("_run_is_gap") == 1) & (pl.col("gap_minutes") <= 15)).then(pl.lit("short"))
                 .when((pl.col("_run_is_gap") == 1) & (pl.col("gap_minutes") <= 60)).then(pl.lit("medium"))
                 .when(pl.col("_run_is_gap") == 1).then(pl.lit("long"))
                 .otherwise(None)
                 .alias("gap_type")
            ])
            .drop(["_chg", "_run_id", "_gap", "_run_is_gap", "_run_len"])
    )
    return out