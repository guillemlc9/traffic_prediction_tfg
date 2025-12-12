"""
normalize_data.py
-----------------
Normalització de les dades per LSTM.
Utilitza estadístiques del conjunt d'entrenament per evitar data leakage.
"""

import polars as pl
from typing import Dict, Tuple
import json
from pathlib import Path


class TrafficNormalizer:
    """
    Normalitzador per a dades de trànsit.
    Calcula mean i std del conjunt d'entrenament i les aplica a tots els splits.
    """
    
    def __init__(self):
        self.mean = None
        self.std = None
        self.is_fitted = False
    
    def fit(self, df: pl.DataFrame, split_col: str = "split") -> "TrafficNormalizer":
        """
        Calcula mean i std només del conjunt d'entrenament.
        
        Args:
            df: DataFrame amb columnes 'estatActual' i 'split'
            split_col: Nom de la columna que indica el split (train/val/test)
        
        Returns:
            self per permetre method chaining
        """
        # Filtrar només dades de train
        train_data = df.filter(pl.col(split_col) == "train")
        
        # Calcular mean i std
        stats = train_data.select([
            pl.col("estatActual").mean().alias("mean"),
            pl.col("estatActual").std().alias("std")
        ]).row(0)
        
        self.mean = stats[0]
        self.std = stats[1]
        self.is_fitted = True
        
        print(f"Normalitzador entrenat:")
        print(f"  Mean (train): {self.mean:.4f}")
        print(f"  Std (train):  {self.std:.4f}")
        
        return self
    
    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Aplica la normalització (z-score) a estatActual.
        
        Args:
            df: DataFrame amb columna 'estatActual'
        
        Returns:
            DataFrame amb columna 'estatActual_norm' afegida
        """
        if not self.is_fitted:
            raise ValueError("El normalitzador no ha estat entrenat. Crida fit() primer.")
        
        return df.with_columns(
            ((pl.col("estatActual") - self.mean) / self.std).alias("estatActual_norm")
        )
    
    def inverse_transform(self, normalized_values: pl.Series) -> pl.Series:
        """
        Reverteix la normalització per obtenir valors originals.
        
        Args:
            normalized_values: Valors normalitzats
        
        Returns:
            Valors en l'escala original
        """
        if not self.is_fitted:
            raise ValueError("El normalitzador no ha estat entrenat. Crida fit() primer.")
        
        return (normalized_values * self.std) + self.mean
    
    def save_params(self, filepath: str):
        """
        Guarda els paràmetres de normalització en un fitxer JSON.
        
        Args:
            filepath: Ruta on guardar els paràmetres
        """
        if not self.is_fitted:
            raise ValueError("El normalitzador no ha estat entrenat. No hi ha paràmetres per guardar.")
        
        params = {
            "mean": float(self.mean),
            "std": float(self.std)
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(params, f, indent=2)
        
        print(f"Paràmetres guardats a: {filepath}")
    
    def load_params(self, filepath: str):
        """
        Carrega els paràmetres de normalització des d'un fitxer JSON.
        
        Args:
            filepath: Ruta del fitxer amb els paràmetres
        """
        with open(filepath, 'r') as f:
            params = json.load(f)
        
        self.mean = params["mean"]
        self.std = params["std"]
        self.is_fitted = True
        
        print(f"Paràmetres carregats de: {filepath}")
        print(f"  Mean: {self.mean:.4f}")
        print(f"  Std:  {self.std:.4f}")


def normalize_traffic_data(df: pl.DataFrame) -> Tuple[pl.DataFrame, TrafficNormalizer]:
    """
    Funció per normalitzar dades de trànsit.
    
    Args:
        df: DataFrame amb columnes 'estatActual' i 'split'
    
    Returns:
        Tuple amb:
            - DataFrame amb columna 'estatActual_norm' afegida
            - Objecte TrafficNormalizer entrenat
    """
    # Crear i entrenar normalitzador
    normalizer = TrafficNormalizer()
    normalizer.fit(df)
    
    # Aplicar transformació
    df_normalized = normalizer.transform(df)
    
    # Mostrar estadístiques per split
    print("\nEstadístiques per split (després de normalització):")
    stats_by_split = df_normalized.group_by("split").agg([
        pl.col("estatActual_norm").mean().alias("mean_norm"),
        pl.col("estatActual_norm").std().alias("std_norm"),
        pl.col("estatActual_norm").min().alias("min_norm"),
        pl.col("estatActual_norm").max().alias("max_norm"),
        pl.count().alias("count")
    ]).sort("split")
    
    print(stats_by_split)
    
    return df_normalized, normalizer


def main():
    """
    Exemple d'ús del normalitzador.
    """
    import sys
    from pathlib import Path
    
    # Afegir el directori arrel al PYTHONPATH
    root_dir = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(root_dir))
    
    from src.lstm.prepare_data import get_all_trams_data
    
    # Carregar dades
    DATA_PATH = "data/processed/dataset_imputed_clean.parquet"
    print(f"Carregant dades de {DATA_PATH}...\n")
    df = pl.read_parquet(DATA_PATH)
    
    # Preparar dades amb splits
    info = get_all_trams_data(df)
    df_with_splits = info["df"]
    
    print(f"Shape original: {df_with_splits.shape}")
    print(f"Columnes: {df_with_splits.columns}\n")
    
    # Normalitzar
    df_normalized, normalizer = normalize_traffic_data(df_with_splits)
    
    print(f"\nShape després de normalització: {df_normalized.shape}")
    print(f"Columnes: {df_normalized.columns}\n")
    
    # Exemple: guardar paràmetres
    params_path = "models/lstm/normalization_params.json"
    normalizer.save_params(params_path)
    
    # Exemple: test inverse transform
    print("\n--- Test Inverse Transform ---")
    sample = df_normalized.filter(pl.col("split") == "test").head(5)
    print("Mostra de dades test:")
    print(sample.select(["idTram", "timestamp", "estatActual", "estatActual_norm"]))
    
    # Revertir normalització
    original_recovered = normalizer.inverse_transform(sample["estatActual_norm"])
    print("\nValors originals recuperats:")
    print(original_recovered)
    print("\nValors originals reals:")
    print(sample["estatActual"])


if __name__ == "__main__":
    main()
