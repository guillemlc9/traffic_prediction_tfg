"""
Mòdul on calcularem per cada tram el seu valor de variànça i el percentatge de valors imputats.
A partir d'aquestes variables, seleccionarem els trams més importants per a la prediccio.
"""

import polars as pl

ruta = "data/processed/dataset_imputed_clean.parquet"
df = pl.read_parquet(ruta)

# Calculem la variànça per cada tram i el percentatge de valors imputats
variance = df.group_by("idTram").agg([
    pl.col("estatActual").var().alias("variancia"),
    (pl.col("is_gap").sum() / pl.len() * 100).alias("pct_imputats")
])

# Ordenem els trams per variància de major a menor
variance = variance.sort(pl.col("variancia"), descending=True)

# Calculem els tercils de la variància (exclou els trams amb variància 0 o null)
variance_non_zero = variance.filter(
    (pl.col("variancia") > 0.0) & (pl.col("variancia").is_not_null())
)

# Calculem els percentils 33 i 66 per dividir en tercils
tercil_33 = variance_non_zero.select(pl.col("variancia").quantile(0.33)).item()
tercil_66 = variance_non_zero.select(pl.col("variancia").quantile(0.66)).item()

print(f"Tercil 33%: {tercil_33:.6f}")
print(f"Tercil 66%: {tercil_66:.6f}")

# Classifiquem els trams segons la variància
variance_classified = variance.with_columns(
    pl.when(pl.col("variancia") == 0.0)
    .then(pl.lit("nul·la"))
    .when(pl.col("variancia") <= tercil_33)
    .then(pl.lit("baixa"))
    .when(pl.col("variancia") <= tercil_66)
    .then(pl.lit("mitjana"))
    .otherwise(pl.lit("alta"))
    .alias("categoria_variancia")
)

# Guardem el DataFrame amb les categories en un fitxer Excel
output_path = "data/processed/trams_variancia_classificats.xlsx"

# Ordenem per categoria i després per variància dins de cada categoria
variance_classified_sorted = variance_classified.sort(
    ["categoria_variancia", "variancia"], 
    descending=[False, True]
)

# Guardem a Excel
variance_classified_sorted.write_excel(output_path)

print(f"\n✅ Fitxer guardat a: {output_path}")
print(f"Total de trams exportats: {len(variance_classified_sorted)}")
