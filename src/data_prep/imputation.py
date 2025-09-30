"""
imputation.py
-------------
Fitxer per imputar els valors no òptims, depenent de la durada de cada
tipus de gap entre registres, amb l'ajuda de gaps.py.
"""

from datetime import timedelta
from pathlib import Path

import polars as pl
from src.data_prep.gaps import find_gaps, classify_gaps

FREQ_MINUTES = 5
freq = timedelta(minutes=FREQ_MINUTES)
freq_interval = f"{FREQ_MINUTES}m"
df = pl.read_parquet("/Users/guillemlopezcolomer/Desktop/traffic_prediction_tfg/data/dataset_1y.parquet")

df = df.with_columns(
    pl.col("timestamp").dt.truncate("5m")
)

gaps = classify_gaps(find_gaps(df))

# 1. Calendari complet (per tram)
cal = (
    df.group_by("idTram")
      .agg(
          pl.min("timestamp").alias("start"),
          pl.max("timestamp").alias("end")
      )
      .with_columns([
          pl.col("start").dt.truncate("5m").alias("start"),
          pl.when(pl.col("end").dt.truncate("5m") == pl.col("end"))
            .then(pl.col("end"))
            .otherwise(pl.col("end").dt.truncate("5m") + freq)
            .alias("end"),
          ((pl.col("end") - pl.col("start"))
             .dt.total_minutes() // FREQ_MINUTES).cast(pl.Int64).alias("n_offsets")
      ])
      .with_columns(pl.int_ranges(0, pl.col("n_offsets") + 1).alias("offset_idx"))
      .explode("offset_idx")
      .with_columns(
          (pl.col("start") + pl.col("offset_idx") * freq).alias("timestamp")
      )
      .select(["idTram", "timestamp"])
)

# 2. Dades + buits
full_df = (
    cal.join(df, on=["idTram", "timestamp"], how="left")
       .with_columns(pl.col("estatActual").is_null().alias("is_gap"))
)

# 3. Etiquetar tipus de gap per cada timestamp buit
expanded_gaps = (
    gaps.with_columns([
        (pl.col("gap_minutes") // FREQ_MINUTES).cast(pl.Int64).alias("n_slots"),
        pl.when(pl.col("gap_minutes") % FREQ_MINUTES == 0)
          .then(pl.col("gap_minutes") // FREQ_MINUTES)
          .otherwise(pl.col("gap_minutes") // FREQ_MINUTES + 1)
          .alias("slots")
    ])
    .with_columns(
        pl.int_ranges(1, pl.col("slots") + 1).alias("offset_idx")
    )
    .explode("offset_idx")
    .with_columns(
        (pl.col("gap_start") + pl.col("offset_idx") * freq).alias("timestamp")
    )
    .select(["idTram", "timestamp", "gap_type"])
)


full_df = full_df.join(expanded_gaps, on=["idTram", "timestamp"], how="left")

full_df = full_df.with_row_index("row_id")

# 4. Imputació per type = short
def impute_short(df_slice):
    df_sorted = df_slice.sort(["idTram", "timestamp"])
    valid = df_sorted.filter(~pl.col("is_gap")).select(["idTram", "timestamp", "estatActual"])

    valid_prev = valid.rename({
        "timestamp": "timestamp_prev",
        "estatActual": "prev"
    })

    prev = df_sorted.join_asof(
        valid_prev,
        left_on="timestamp",
        right_on="timestamp_prev",
        by="idTram",
        strategy="backward"
    )

    prev = prev.select([
        "row_id",
        "idTram",
        "timestamp",
        "estatActual",
        "is_gap",
        "gap_type",
        "prev",
        "timestamp_prev"
    ])

    valid_next = valid.rename({
        "timestamp": "timestamp_next",
        "estatActual": "next"
    })

    nxt = df_sorted.join_asof(
        valid_next,
        left_on="timestamp",
        right_on="timestamp_next",
        by="idTram",
        strategy="forward"
    )

    nxt = nxt.select([
        "row_id",
        "next",
        "timestamp_next"
    ])

    ts = pl.col("timestamp").cast(pl.Int64)
    ts_prev = pl.col("timestamp_prev").cast(pl.Int64)
    ts_next = pl.col("timestamp_next").cast(pl.Int64)

    ratio = (
        pl.when(
            (pl.col("prev").is_null()) |
            (pl.col("next").is_null()) |
            (ts_next - ts_prev == 0) |
            ts_prev.is_null() |
            ts_next.is_null()
        )
          .then(pl.lit(None))
          .otherwise((ts - ts_prev).cast(pl.Float64) / (ts_next - ts_prev))
    )

    interp = (
        pl.col("prev") + (pl.col("next") - pl.col("prev")) * ratio
    ).round().cast(pl.Int64)

    return (
        prev.join(nxt, on="row_id", how="left")
            .with_columns(
                pl.when(pl.col("is_gap") & (pl.col("gap_type") == "short"))
                  .then(
                      pl.when(pl.col("prev").is_not_null() & pl.col("next").is_not_null() & ratio.is_not_null())
                        .then(interp)
                        .when(pl.col("prev").is_not_null())
                        .then(pl.col("prev").cast(pl.Int64))
                        .when(pl.col("next").is_not_null())
                        .then(pl.col("next").cast(pl.Int64))
                        .otherwise(pl.col("estatActual"))
                  )
                  .otherwise(pl.col("estatActual"))
                  .alias("estatActual")
            )
    )

# 5. Imputació per type = medium (rolling mode)
def impute_medium(df_slice: pl.DataFrame, window_minutes: int = 60) -> pl.DataFrame:
    window_ns = window_minutes * 60 * 1_000_000_000

    valid = df_slice.filter(~pl.col("estatActual").is_null()).select(
        ["idTram", "timestamp", "estatActual"]
    )
    medium = df_slice.filter(
        pl.col("is_gap") & (pl.col("gap_type") == "medium")
    ).select(["row_id", "idTram", "timestamp"])

    if medium.is_empty() or valid.is_empty():
        return df_slice

    candidates = (
        medium.join(valid, on="idTram", how="inner", suffix="_valid")
              .with_columns(
                  (pl.col("timestamp_valid") - pl.col("timestamp"))
                  .dt.nanosecond()
                  .abs()
                  .alias("diff_ns")
              )
              .filter(pl.col("diff_ns") <= window_ns)
    )

    if candidates.is_empty():
        return df_slice

    imputed = (
        candidates
        .group_by(["row_id", "estatActual_valid"])
        .agg(pl.len().alias("weight"))
        .group_by("row_id")
        .agg(
            pl.col("estatActual_valid")
              .take(pl.col("weight").arg_max())
              .alias("estatActual_imputed")
        )
    )

    return (
        df_slice.join(imputed, on="row_id", how="left")
                .with_columns(
                    pl.when(pl.col("estatActual").is_null() & pl.col("estatActual_imputed").is_not_null())
                      .then(pl.col("estatActual_imputed").cast(pl.Int64))
                      .otherwise(pl.col("estatActual"))
                      .alias("estatActual")
                )
                .drop("estatActual_imputed")
    )

# Exemple: aplica estratègies short i medium
full_df = impute_short(full_df)
full_df = impute_medium(full_df)

# 6. Export
full_df = full_df.with_columns(
    pl.col("estatActual").is_null().not_().alias("estatActual_imputed")
).drop(["row_id", "prev", "next", "timestamp_prev", "timestamp_next"], strict=False)
Path("data/processed").mkdir(parents=True, exist_ok=True)
full_df.write_parquet("data/processed/dataset_imputed.parquet")
