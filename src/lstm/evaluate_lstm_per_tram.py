"""
evaluate_lstm_per_tram.py
--------------------------
Avalua el model LSTM global calculant mètriques (MAE, RMSE) per cada tram individual.

Aquest script:
1. Carrega les prediccions desnormalitzades del model LSTM
2. Reconstrueix la correspondència entre prediccions i trams
3. Calcula MAE i RMSE per cada tram dels 30 seleccionats
4. Guarda els resultats en un fitxer parquet
"""

import sys
from pathlib import Path

# Afegir el directori arrel al PYTHONPATH
root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root_dir))

import polars as pl
import numpy as np
from typing import Dict, List
from src.lstm.prepare_data import get_all_trams_data
from src.lstm.normalize_data import normalize_traffic_data
from src.data_prep.prepare_time_splits import SELECTED_TRAMS


def create_tram_mapping_for_test(
    df_normalized: pl.DataFrame,
    window_size: int = 36,
    horizon: int = 1,
    value_col: str = "estatActual_norm"
) -> Dict[int, List[int]]:
    """
    Reconstrueix el mapatge entre índexs de prediccions test i trams.
    
    Aquesta funció replica la lògica de create_all_sequences() però només
    per al conjunt de test, i retorna quin tram correspon a cada predicció.
    
    Parameters
    ----------
    df_normalized : pl.DataFrame
        DataFrame normalitzat amb columna 'split'
    window_size : int
        Mida de la finestra temporal
    horizon : int
        Horitzó de predicció
    value_col : str
        Nom de la columna amb valors normalitzats
    
    Returns
    -------
    dict
        Diccionari amb:
        - 'tram_ids': llista amb l'ID del tram per cada predicció test
        - 'indices': llista amb l'índex dins del tram
        - 'n_test': nombre total de prediccions test
    """
    # Ordenar igual que en create_all_sequences
    df = df_normalized.sort(["idTram", "timestamp"])
    
    tram_ids_list = []
    indices_list = []
    test_idx = 0
    
    # Iterar per cada tram
    for tram_id in SELECTED_TRAMS:
        # Filtrar dades d'aquest tram
        tram_df = df.filter(pl.col("idTram") == tram_id)
        
        if tram_df.height == 0:
            continue
        
        # Extreure arrays
        v = tram_df[value_col].to_numpy().astype("float32")
        s = tram_df["split"].to_list()
        
        n = len(v)
        max_start = n - window_size - horizon + 1
        if max_start <= 0:
            print(f"  Tram {tram_id} ignorat (massa pocs punts: {n})")
            continue
        
        for start in range(max_start):
            end = start + window_size
            target_idx = end + horizon - 1
            split_target = s[target_idx]
            
            if split_target == "test":
                tram_ids_list.append(tram_id)
                indices_list.append(target_idx)
                test_idx += 1
    
    return {
        'tram_ids': tram_ids_list,
        'indices': indices_list,
        'n_test': len(tram_ids_list)
    }


def calculate_metrics_per_tram(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    tram_mapping: Dict
) -> pl.DataFrame:
    """
    Calcula MAE i RMSE per cada tram.
    
    Parameters
    ----------
    y_test : np.ndarray
        Valors reals (desnormalitzats)
    y_pred : np.ndarray
        Prediccions (desnormalitzades)
    tram_mapping : dict
        Mapatge entre índexs i trams
    
    Returns
    -------
    pl.DataFrame
        DataFrame amb columnes: idTram, mae, rmse, n_samples
    """
    tram_ids = np.array(tram_mapping['tram_ids'])
    
    results = []
    for tram_id in SELECTED_TRAMS:
        # Filtrar prediccions d'aquest tram
        mask = tram_ids == tram_id
        
        if not mask.any():
            print(f"  ⚠️  Tram {tram_id}: No hi ha prediccions test")
            continue
        
        y_true_tram = y_test[mask]
        y_pred_tram = y_pred[mask]
        
        # Calcular mètriques
        mae = float(np.mean(np.abs(y_true_tram - y_pred_tram)))
        mse = float(np.mean((y_true_tram - y_pred_tram) ** 2))
        rmse = float(np.sqrt(mse))
        
        results.append({
            'idTram': tram_id,
            'mae': mae,
            'rmse': rmse,
            'mse': mse,
            'n_samples': int(mask.sum())
        })
        
        print(f"  Tram {tram_id:3d}: MAE={mae:.4f}, RMSE={rmse:.4f}, n={mask.sum()}")
    
    return pl.DataFrame(results)


def main():
    """
    Script principal per avaluar LSTM per tram.
    """
    # Configuració
    MODEL_DIR = "models/lstm/20251227_143405_lr1e-03_bs1024_u64_w36_h1"
    DATA_PATH = "data/processed/dataset_imputed_clean.parquet"
    WINDOW_SIZE = 36
    HORIZON = 1
    
    print("=" * 60)
    print("AVALUACIÓ LSTM PER TRAM")
    print("=" * 60)
    print(f"Model: {MODEL_DIR}")
    print(f"Trams seleccionats: {len(SELECTED_TRAMS)}")
    print("=" * 60)
    
    # 1. Carregar prediccions desnormalitzades
    print("\n1. Carregant prediccions desnormalitzades...")
    model_path = Path(MODEL_DIR)
    denorm_path = model_path / "denormalized"
    
    y_test_denorm = np.load(denorm_path / "y_test_denorm.npy")
    y_pred_denorm = np.load(denorm_path / "y_pred_denorm.npy")
    
    print(f"   ✓ y_test shape: {y_test_denorm.shape}")
    print(f"   ✓ y_pred shape: {y_pred_denorm.shape}")
    
    # 2. Carregar i preparar dades per reconstruir el mapatge
    print("\n2. Reconstruint mapatge tram-predicció...")
    df = pl.read_parquet(DATA_PATH)
    info = get_all_trams_data(df)
    df_with_splits = info["df"]
    df_normalized, normalizer = normalize_traffic_data(df_with_splits)
    
    # 3. Crear mapatge
    print("\n3. Creant mapatge per conjunt test...")
    tram_mapping = create_tram_mapping_for_test(
        df_normalized,
        window_size=WINDOW_SIZE,
        horizon=HORIZON,
        value_col="estatActual_norm"
    )
    
    print(f"   ✓ Total prediccions test: {tram_mapping['n_test']}")
    
    # Verificar que coincideix amb les prediccions carregades
    if tram_mapping['n_test'] != len(y_test_denorm):
        print(f"\n   ⚠️  ADVERTÈNCIA: Desajust en nombre de prediccions!")
        print(f"      Mapatge: {tram_mapping['n_test']}")
        print(f"      Carregades: {len(y_test_denorm)}")
        raise ValueError("El nombre de prediccions no coincideix amb el mapatge")
    
    # 4. Calcular mètriques per tram
    print("\n4. Calculant mètriques per tram...")
    metrics_df = calculate_metrics_per_tram(
        y_test_denorm,
        y_pred_denorm,
        tram_mapping
    )
    
    # 5. Guardar resultats
    output_path = model_path / "metrics_per_tram.parquet"
    metrics_df.write_parquet(output_path)
    print(f"\n✅ Mètriques guardades: {output_path}")
    
    # 6. Mostrar resum
    print("\n" + "=" * 60)
    print("RESUM MÈTRIQUES LSTM PER TRAM")
    print("=" * 60)
    print(f"\nTrams avaluats: {metrics_df.height}")
    print(f"\nMètriques mitjanes:")
    print(f"  MAE mitjà:  {metrics_df['mae'].mean():.4f}")
    print(f"  RMSE mitjà: {metrics_df['rmse'].mean():.4f}")
    
    print(f"\nDistribució MAE:")
    print(f"  Min:     {metrics_df['mae'].min():.4f}")
    print(f"  Q1:      {metrics_df['mae'].quantile(0.25):.4f}")
    print(f"  Mediana: {metrics_df['mae'].median():.4f}")
    print(f"  Q3:      {metrics_df['mae'].quantile(0.75):.4f}")
    print(f"  Max:     {metrics_df['mae'].max():.4f}")
    
    print(f"\nDistribució RMSE:")
    print(f"  Min:     {metrics_df['rmse'].min():.4f}")
    print(f"  Q1:      {metrics_df['rmse'].quantile(0.25):.4f}")
    print(f"  Mediana: {metrics_df['rmse'].median():.4f}")
    print(f"  Q3:      {metrics_df['rmse'].quantile(0.75):.4f}")
    print(f"  Max:     {metrics_df['rmse'].max():.4f}")
    
    print("\n" + "=" * 60)
    print("✅ Avaluació completada!")
    print("=" * 60)


if __name__ == "__main__":
    main()
