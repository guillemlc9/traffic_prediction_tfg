"""
imputation.py
-------------
Fitxer per imputar els valors no òptims, depenent de la durada de cada
tipus de gap entre registres, amb l'ajuda de gaps.py.
"""

from pathlib import Path
import os
from typing import List
from datetime import timedelta

import polars as pl
from src.data_prep.gaps import find_gaps, classify_gaps, classify_gaps_from_calendar

FREQ_MINUTES = 5
FREQ = pl.duration(minutes=FREQ_MINUTES)
LOOKBACK_STEPS = 12

SRC_PARQUET = "/Users/guillemlopezcolomer/Desktop/traffic_prediction_tfg/data/dataset_1y.parquet"
OUT_DIR = "data/processed"
OUT_PARQUET = f"{OUT_DIR}/dataset_imputed.parquet"

# ---------- Utils ----------

def snap_to_5min_mode(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df
        .with_columns(
            pl.when(pl.col("estatActual").is_in([0, 6]))
              .then(None)
              .otherwise(pl.col("estatActual"))
              .alias("estatActual")
        )
        .with_columns(pl.col("timestamp").dt.truncate(f"{FREQ_MINUTES}m").alias("timestamp_5m"))
        .group_by(["idTram", "timestamp_5m"])
        .agg([
            pl.col("estatActual").mode().first().alias("estatActual_mode"),
            pl.col("estatActual").last().alias("estatActual_last"),
        ])
        .with_columns(
            pl.when(pl.col("estatActual_mode").is_not_null())
              .then(pl.col("estatActual_mode"))
              .otherwise(pl.col("estatActual_last"))
              .cast(pl.Int8)
              .alias("estatActual")
        )
        .rename({"timestamp_5m": "timestamp"})
        .select(["idTram", "timestamp", "estatActual"])
        .sort(["idTram", "timestamp"])
    )

def build_calendar_for_tram(df5_tram: pl.DataFrame) -> pl.DataFrame:
    """
    Calendari complet (graella 5') entre min i max per a UN idTram.
    """
    start = df5_tram["timestamp"].min()
    end   = df5_tram["timestamp"].max()

    F = timedelta(minutes=FREQ_MINUTES)

    def truncate_5m(dt):
        m = (dt.minute // FREQ_MINUTES) * FREQ_MINUTES
        return dt.replace(minute=m, second=0, microsecond=0)

    start_tr = truncate_5m(start)
    end_tr   = truncate_5m(end)
    end_adj  = end_tr if (end == end_tr) else (end_tr + F)

    n_steps = int((end_adj - start_tr) / F)
    ts_list = [start_tr + i * F for i in range(n_steps + 1)]

    return pl.DataFrame({"timestamp": ts_list})

def impute_short(df_slice: pl.DataFrame) -> pl.DataFrame:
    valid = df_slice.filter(~pl.col("estatActual").is_null()).select(["timestamp","estatActual"])

    valid_prev = valid.rename({"timestamp":"timestamp_prev","estatActual":"prev"}).sort("timestamp_prev")
    prev = df_slice.join_asof(
        valid_prev, left_on="timestamp", right_on="timestamp_prev",
        strategy="backward"
    )

    valid_next = valid.rename({"timestamp":"timestamp_next","estatActual":"next"}).sort("timestamp_next")
    nxt = prev.join_asof(
        valid_next, left_on="timestamp", right_on="timestamp_next",
        strategy="forward"
    )

    ts      = pl.col("timestamp").cast(pl.Int64)
    ts_prev = pl.col("timestamp_prev").cast(pl.Int64)
    ts_next = pl.col("timestamp_next").cast(pl.Int64)

    ratio = pl.when(
        pl.any_horizontal([
            pl.col("prev").is_null(),
            pl.col("next").is_null(),
            (ts_next - ts_prev == 0),
            ts_prev.is_null(),
            ts_next.is_null(),
        ])
    ).then(pl.lit(None)).otherwise((ts - ts_prev).cast(pl.Float64) / (ts_next - ts_prev))

    interp = (pl.col("prev") + (pl.col("next") - pl.col("prev")) * ratio).round().cast(pl.Int8)

    return (
        nxt.with_columns(
            pl.when(pl.col("is_gap") & (pl.col("gap_type") == "short"))
              .then(
                  pl.when(pl.all_horizontal([pl.col("prev").is_not_null(),
                                             pl.col("next").is_not_null(),
                                             ratio.is_not_null()]))
                   .then(interp)
                   .when(pl.col("prev").is_not_null()).then(pl.col("prev").cast(pl.Int8))
                   .when(pl.col("next").is_not_null()).then(pl.col("next").cast(pl.Int8))
                   .otherwise(pl.col("estatActual"))
              )
              .otherwise(pl.col("estatActual"))
              .alias("estatActual")
        )
        .drop(["prev","next","timestamp_prev","timestamp_next"])
    )

def impute_medium(df_slice: pl.DataFrame, window_minutes: int = 60) -> pl.DataFrame:
    """
    Medium gaps: pren el valor vàlid més proper dins ±window_minutes.
    """
    tol = timedelta(minutes=window_minutes)

    valid = (
        df_slice
        .filter(~pl.col("estatActual").is_null())
        .select(["timestamp", "estatActual"])
        .sort("timestamp")
    )
    medix = (
        df_slice
        .filter(pl.col("is_gap") & (pl.col("gap_type") == "medium"))
        .select(["timestamp"])
        .sort("timestamp")
    )

    if medix.is_empty() or valid.is_empty():
        return df_slice

    # Veí anterior dins tolerància
    back = medix.join_asof(
        valid.rename({"timestamp": "ts_v", "estatActual": "val_b"}),
        left_on="timestamp",
        right_on="ts_v",
        strategy="backward",
        tolerance=tol,
    )

    # Veí posterior dins tolerància
    fwd = medix.join_asof(
        valid.rename({"timestamp": "ts_v", "estatActual": "val_f"}),
        left_on="timestamp",
        right_on="ts_v",
        strategy="forward",
        tolerance=tol,
    )

    # Combina escollint el més proper (Polars 0.20.29+ → how="full")
    both = (
        back.join(fwd, on="timestamp", how="full")
            .with_columns([
                (pl.col("timestamp") - pl.col("ts_v")).abs().alias("d_b"),
                (pl.col("timestamp") - pl.col("ts_v_right")).abs().alias("d_f"),
            ])
            .with_columns(
                pl.when(pl.col("val_b").is_null() & pl.col("val_f").is_not_null()).then(pl.col("val_f"))
                .when(pl.col("val_f").is_null() & pl.col("val_b").is_not_null()).then(pl.col("val_b"))
                .when(pl.col("val_b").is_not_null() & pl.col("val_f").is_not_null())
                    .then(pl.when(pl.col("d_b") <= pl.col("d_f")).then(pl.col("val_b")).otherwise(pl.col("val_f")))
                .otherwise(None)
                .alias("estatActual_imputed")
            )
            .select(["timestamp", "estatActual_imputed"])
    )

    return (
        df_slice.join(both, on="timestamp", how="left")
                .with_columns(
                    pl.when(pl.col("estatActual").is_null() & pl.col("estatActual_imputed").is_not_null())
                      .then(pl.col("estatActual_imputed").cast(pl.Int8))
                      .otherwise(pl.col("estatActual"))
                      .alias("estatActual")
                )
                .drop("estatActual_imputed")
    )

def impute_long_with_history(df_slice: pl.DataFrame) -> pl.DataFrame:
    """
    Long gaps: imputa amb el MODE històric per (weekday, slot_5min).
    - weekday: 0=dl ... 6=dg
    - slot_5min: minut-del-dia // 5
    Fa servir NOMÉS punts no-gap (originals) per construir la història.
    """
    # features temporals
    tmp = df_slice.with_columns([
        pl.col("timestamp").dt.weekday().alias("_wd"),
        (
            pl.col("timestamp").dt.hour() * 12
            + (pl.col("timestamp").dt.minute() / 5).floor()
        ).cast(pl.Int16).alias("_slot"),
    ])

    # Històric: comptem freqüències per (wd, slot, estatActual) i triem el més freqüent
    hist = (
        tmp.filter(~pl.col("is_gap") & pl.col("estatActual").is_not_null())
           .group_by(["_wd", "_slot", "estatActual"])
           .agg(pl.len().alias("w"))
           .sort(["_wd", "_slot", "w"], descending=[False, False, True])
           .group_by(["_wd", "_slot"])
           .agg(pl.col("estatActual").first().alias("_mode_hist"))
    )

    # Fallback global (per si algun (wd,slot) no té prou històric)
    fb_tbl = (
        tmp.filter(~pl.col("is_gap") & pl.col("estatActual").is_not_null())
           .group_by("estatActual").agg(pl.len().alias("w"))
           .sort("w", descending=True)
           .select(pl.col("estatActual").first().alias("_global_mode"))
    )
    global_mode = fb_tbl["_global_mode"][0] if fb_tbl.height > 0 else None

    # Aplica només sobre long
    out = (
        tmp.join(hist, on=["_wd", "_slot"], how="left")
           .with_columns(
               pl.when(
                   pl.col("estatActual").is_null()
                   & pl.col("is_gap")
                   & (pl.col("gap_type") == "long")
                   & pl.col("_mode_hist").is_not_null()
               )
               .then(pl.col("_mode_hist").cast(pl.Int8))
               .when(
                   pl.col("estatActual").is_null()
                   & pl.col("is_gap")
                   & (pl.col("gap_type") == "long")
                   & pl.lit(global_mode).is_not_null()
               )
               .then(pl.lit(global_mode).cast(pl.Int8))
               .otherwise(pl.col("estatActual"))
               .alias("estatActual")
           )
           .drop(["_wd", "_slot", "_mode_hist"])
    )
    return out

# ---------- Main ----------

def main() -> None:
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    if os.path.exists(OUT_PARQUET):
        os.remove(OUT_PARQUET)

    # Llegeix i redueix dtypes
    df = pl.read_parquet(SRC_PARQUET).select([
        pl.col("idTram").cast(pl.Int32),
        pl.col("timestamp").cast(pl.Datetime),
        pl.col("estatActual").cast(pl.Int8),
        pl.col("estatPrevist").cast(pl.Int8, strict=False)
    ]).sort(["idTram","timestamp"])

    df = df.with_columns(
        pl.when(pl.col("estatActual").is_in([0, 6]))
        .then(None)
        .otherwise(pl.col("estatActual"))
        .cast(pl.Int8)
        .alias("estatActual")
    )

    # Snap a 5'
    trams: List[int] = df.select("idTram").unique().to_series().to_list()
    all_chunks: List[pl.DataFrame] = []

    for i, t in enumerate(trams, 1):
        df_t = df.filter(pl.col("idTram") == t).select(["idTram","timestamp","estatActual"])
        if df_t.is_empty():
            continue

        df5_t = snap_to_5min_mode(df_t)
        cal_t = build_calendar_for_tram(df5_t).with_columns(pl.lit(t).cast(pl.Int32).alias("idTram"))

        full_t = (
            cal_t.join(df5_t, on=["idTram","timestamp"], how="left")
                 .with_columns(pl.col("estatActual").is_null().alias("is_gap"))
                 .sort(["timestamp"])
        )

        # Gaps per tram
        full_t = classify_gaps_from_calendar(full_t, freq_minutes=FREQ_MINUTES)

        # Imputacions
        full_t = impute_short(full_t)
        full_t = impute_medium(full_t, window_minutes=60)
        full_t = impute_long_with_history(full_t)

        # Bandera final i selecció de columnes
        full_t = (
            full_t.with_columns(pl.col("estatActual").is_null().not_().alias("estatActual_imputed"))
                  .select(["idTram","timestamp","estatActual","is_gap","gap_type","estatActual_imputed"])
        )

        all_chunks.append(full_t)

        if i % 10 == 0 or i == len(trams):
            print(f"[{i}/{len(trams)}] Processat idTram={t} -> {full_t.height} files")

    # Escriure un únic Parquet (sense append)
    if all_chunks:
        combined = pl.concat(all_chunks, how="vertical_relaxed")
        combined.write_parquet(OUT_PARQUET)

        OUT_PARQUET_CLEAN = f"{OUT_DIR}/dataset_imputed_clean.parquet"
        combined_clean = combined.filter(pl.col("estatActual").is_between(1, 5))
        combined_clean.write_parquet(OUT_PARQUET_CLEAN)

        out = combined.select([
            pl.len().alias("rows"),
            pl.col("estatActual").is_null().sum().alias("nulls_estatActual"),
            pl.col("is_gap").sum().alias("gaps_detectats"),
            pl.col("estatActual_imputed").sum().alias("valors_no_null_despres_imputacio"),
        ])
        print(out)
        print(f"✔ Parquet complet: {OUT_PARQUET}")
        print(f"✔ Parquet net (1..5): {OUT_PARQUET_CLEAN}")
    else:
        print("No s'han generat chunks (cap tram amb dades).")

if __name__ == "__main__":
    main()