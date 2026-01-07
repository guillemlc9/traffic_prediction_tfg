"""
prepare_time_splits.py
----------------------
Prepara les sèries temporals dels 30 trams seleccionats amb splits temporals
consistents (75% train, 10% val, 15% test) per a tots els models.
"""

import polars as pl
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple


# Els 30 trams seleccionats
SELECTED_TRAMS = [
    233, 158, 57, 526, 83, 445, 11, 22, 460, 178, 
    126, 534, 164, 50, 278, 388, 478, 270, 332, 254, 
    224, 293, 117, 100, 104, 409, 31, 221, 360, 303
]


def get_temporal_splits(df: pl.DataFrame) -> Dict[str, datetime]:
    """
    Calcula les dates de tall per train/val/test basant-se en el rang temporal complet.
    
    Splits:
    - Train: primers 75% dels mesos
    - Validation: següent 10% dels mesos
    - Test: últims 15% dels mesos
    
    Parameters
    ----------
    df : pl.DataFrame
        DataFrame amb columna 'timestamp'
    
    Returns
    -------
    dict
        Diccionari amb les dates de tall:
        {
            'train_start': datetime,
            'train_end': datetime,
            'val_start': datetime,
            'val_end': datetime,
            'test_start': datetime,
            'test_end': datetime
        }
    """
    # Rang temporal complet
    start_date = df['timestamp'].min()
    end_date = df['timestamp'].max()
    
    # Durada total
    total_duration = end_date - start_date
    
    # Punts de tall
    train_end = start_date + total_duration * 0.75
    val_end = start_date + total_duration * 0.85
    
    splits = {
        'train_start': start_date,
        'train_end': train_end,
        'val_start': train_end,
        'val_end': val_end,
        'test_start': val_end,
        'test_end': end_date
    }
    
    return splits


def prepare_tram_series(
    df: pl.DataFrame, 
    tram_id: int, 
    split: str,
    splits_info: Dict[str, datetime] = None
) -> pd.DataFrame:
    """
    Retorna la sèrie temporal d'un tram per un split específic.
    
    Parameters
    ----------
    df : pl.DataFrame
        DataFrame complet amb totes les dades
    tram_id : int
        ID del tram a extreure
    split : str
        Split desitjat: 'train', 'val' o 'test'
    splits_info : dict, optional
        Diccionari amb les dates de tall. Si None, es calcula automàticament.
    
    Returns
    -------
    pd.DataFrame
        Sèrie temporal del tram amb index temporal i columna 'estatActual'
    """
    if splits_info is None:
        splits_info = get_temporal_splits(df)
    
    # Determinar rang temporal segons el split
    if split == 'train':
        start = splits_info['train_start']
        end = splits_info['train_end']
    elif split == 'val':
        start = splits_info['val_start']
        end = splits_info['val_end']
    elif split == 'test':
        start = splits_info['test_start']
        end = splits_info['test_end']
    else:
        raise ValueError(f"Split '{split}' no vàlid. Utilitza 'train', 'val' o 'test'")
    
    # Filtrar per tram i rang temporal
    df_filtered = (
        df.filter(
            (pl.col('idTram') == tram_id) &
            (pl.col('timestamp') >= start) &
            (pl.col('timestamp') <= end)
        )
        .select(['timestamp', 'estatActual'])
        .sort('timestamp')
        .to_pandas()
    )
    
    # Establir timestamp com a index
    df_filtered = df_filtered.set_index('timestamp')
    
    return df_filtered


def get_all_trams_data(
    df: pl.DataFrame, 
    tram_ids: List[int] = None
) -> Dict[str, any]:
    """
    Prepara totes les sèries amb informació dels splits.
    
    Parameters
    ----------
    df : pl.DataFrame
        DataFrame complet amb totes les dades
    tram_ids : list, optional
        Llista d'IDs de trams. Si None, utilitza SELECTED_TRAMS.
    
    Returns
    -------
    dict
        Diccionari amb:
        - 'splits': informació de les dates de tall
        - 'tram_ids': llista d'IDs de trams
        - Funcions helper per accedir a les dades
    """
    if tram_ids is None:
        tram_ids = SELECTED_TRAMS
    
    # Splits
    splits_info = get_temporal_splits(df)
    
    # Filtrar dataset pels trams seleccionats
    df_filtered = df.filter(pl.col('idTram').is_in(tram_ids))
    
    return {
        'splits': splits_info,
        'tram_ids': tram_ids,
        'df': df_filtered,
        'get_tram_series': lambda tram_id, split: prepare_tram_series(
            df_filtered, tram_id, split, splits_info
        )
    }


def print_splits_summary(splits_info: Dict[str, datetime]) -> None:
    """
    Mostra un resum dels splits temporals.
    
    Parameters
    ----------
    splits_info : dict
        Diccionari amb les dates de tall
    """
    print("=" * 60)
    print("SPLITS TEMPORALS")
    print("=" * 60)
    
    # Calcular durades
    total_duration = splits_info['test_end'] - splits_info['train_start']
    train_duration = splits_info['train_end'] - splits_info['train_start']
    val_duration = splits_info['val_end'] - splits_info['val_start']
    test_duration = splits_info['test_end'] - splits_info['test_start']
    
    # Calcular percentatges
    train_pct = (train_duration / total_duration) * 100
    val_pct = (val_duration / total_duration) * 100
    test_pct = (test_duration / total_duration) * 100
    
    print(f"\nTRAIN ({train_pct:.1f}%):")
    print(f"  {splits_info['train_start']} → {splits_info['train_end']}")
    print(f"  Durada: {train_duration.days} dies")
    
    print(f"\nVALIDATION ({val_pct:.1f}%):")
    print(f"  {splits_info['val_start']} → {splits_info['val_end']}")
    print(f"  Durada: {val_duration.days} dies")
    
    print(f"\nTEST ({test_pct:.1f}%):")
    print(f"  {splits_info['test_start']} → {splits_info['test_end']}")
    print(f"  Durada: {test_duration.days} dies")
    
    print(f"\nTOTAL:")
    print(f"  {splits_info['train_start']} → {splits_info['test_end']}")
    print(f"  Durada: {total_duration.days} dies")
    print("=" * 60)


def main():
    """
    Ús del mòdul.
    """
    # Carregar dades
    DATA_PATH = "data/processed/dataset_imputed_clean.parquet"
    print(f"Carregant dades de {DATA_PATH}...")
    df = pl.read_parquet(DATA_PATH)
    
    # Informació dels splits
    splits_info = get_temporal_splits(df)
    print_splits_summary(splits_info)
    
    # Preparar dades de tots els trams
    data = get_all_trams_data(df, SELECTED_TRAMS)
    
    # Obtenim dades d'un tram
    for split in ['train', 'val', 'test']:
        series = data['get_tram_series'](SELECTED_TRAMS[0], split)
        print(f"  {split.upper()}: {len(series)} registres")
        print(f"    Rang: {series.index.min()} → {series.index.max()}")
    
    # Verificar tots els trams
    for tram_id in SELECTED_TRAMS[:5]:  # Mostrar només 5 primers
        train = data['get_tram_series'](tram_id, 'train')
        val = data['get_tram_series'](tram_id, 'val')
        test = data['get_tram_series'](tram_id, 'test')
        total = len(train) + len(val) + len(test)
        print(f"  Tram {tram_id:3d}: Train={len(train):6d}, Val={len(val):5d}, Test={len(test):5d}, Total={total:6d}")
    


if __name__ == "__main__":
    main()
