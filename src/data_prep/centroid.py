"""
centroid.py
-----------
Script que calcula el punt centroid de les coordenades
de cada tram a partir del fitxer de coordenades original.
Genera un fitxer .parquet amb una sola fila per tram.
"""

from pathlib import Path
import pandas as pd

# --- Variables ---
CSV_PATH = "/Users/guillemlopezcolomer/Desktop/traffic_prediction_tfg/data/transit_relacio_trams_format_long.csv"
OUT_DIR = "/Users/guillemlopezcolomer/Desktop/traffic_prediction_tfg/data/processed"
OUT_PARQUET = f"{OUT_DIR}/trams_centroid.parquet"

def main() -> None:
    # Llegeix el CSV
    df = pd.read_csv(CSV_PATH)

    # Tipus numèric a les coordenades
    for col in ["Latitud", "Longitud"]:
        df[col] = pd.to_numeric(df[col], errors = "coerce")
    
    # Calcula el centroid per cada tram
    centroids = (
        df.groupby("Tram", as_index=False)
          .agg({"Latitud": "mean", "Longitud": "mean"})
          .rename(columns= {"Latitud": "lat_centre", "Longitud": "lon_centre"})
    )

    #Desa en format Parquet
    centroids.to_parquet(OUT_PARQUET, index=False)
    print(f"✔ Fitxer Parquet escrit a: {OUT_PARQUET}")
    print(f"{len(centroids)} centroides generats.")

if __name__ == "__main__":
    main()