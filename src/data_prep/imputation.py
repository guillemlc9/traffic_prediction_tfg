import polars as pl
from src.data_prep.gaps import find_gaps, classify_gaps

freq = pl.duration(minutes=5)
df = pl.read_parquet("data/raw/dataset_1y.parquet")

gaps = classify_gaps(find_gaps(df))

# 1. Calendari complet (per tram)
cal = (
    df.group_by("idTram")
      .agg(
          pl.min("timestamp").alias("start"),
          pl.max("timestamp").alias("end")
      )
      .with_columns(
          pl.col("start").dt.truncate("5m"),
          pl.col("end").dt.ceil("5m")
      )
      .explode(
          pl.date_ranges(
              pl.col("start"),
              pl.col("end"),
              interval=freq,
              eager=True
          ).alias("timestamp")
      )
      .select(["idTram", "timestamp"])
)

# 2. Dades + buits
full_df = (
    cal.join(df, on=["idTram", "timestamp"], how="left")
       .with_columns(pl.col("velocitat").is_null().alias("is_gap"))
)

# 3. Etiquetar tipus de gap per cada timestamp buit
expanded_gaps = (
    gaps.with_columns([
        (pl.col("gap_minutes") // 5).cast(pl.Int64).alias("n_slots"),
        pl.when(pl.col("gap_minutes") % 5 == 0)
          .then(pl.col("gap_minutes") // 5)
          .otherwise(pl.col("gap_minutes") // 5 + 1)
          .alias("slots")
    ])
    .with_columns(
        pl.date_ranges(
            pl.col("gap_start") + freq,
            pl.col("gap_end"),
            interval=freq,
            eager=True
        ).alias("missing_ts")
    )
    .explode("missing_ts")
    .select(["idTram", pl.col("missing_ts").alias("timestamp"), "gap_type"])
)

full_df = full_df.join(expanded_gaps, on=["idTram", "timestamp"], how="left")

# 4. Imputació condicionada
def impute_short(df_slice):
    prev = df_slice.join_asof(
        df_slice.filter(~pl.col("is_gap")),
        on="timestamp",
        by="idTram",
        strategy="backward"
    ).rename({"velocitat_right": "prev"})
    nxt = df_slice.join_asof(
        df_slice.filter(~pl.col("is_gap")),
        on="timestamp",
        by="idTram",
        strategy="forward"
    ).rename({"velocitat_right": "next"})
    return (
        prev.join(nxt, on=["idTram", "timestamp"])
            .with_columns(
                pl.when(pl.col("is_gap") & (pl.col("gap_type") == "short"))
                  .then(
                      pl.col("prev") + (pl.col("next") - pl.col("prev")) *
                      ((pl.col("timestamp") - pl.col("timestamp_prev")).dt.nanosecond()
                       / (pl.col("timestamp_next") - pl.col("timestamp_prev")).dt.nanosecond())
                  )
                  .otherwise(pl.col("velocitat"))
                  .alias("velocitat")
            )
    )

# Exemple: aplica estratègia short; implementa de forma similar per medium/long
full_df = impute_short(full_df)

# 5. Bandera i export
full_df = full_df.with_columns(
    pl.col("velocitat").is_null().not_().alias("imputed_flag")
)
full_df.write_parquet("data/processed/dataset_imputed.parquet")
