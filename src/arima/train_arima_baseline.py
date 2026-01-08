"""
train_arima_baseline.py
-----------------------
Entrena 30 models ARIMA(1,1,1) utilitzant només les dades de TRAIN
per als trams seleccionats.

Aquest és el model baseline per comparar amb LSTM i models espai-temporals.
"""

import sys
from pathlib import Path

# Afegim el directori arrel al PYTHONPATH
root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root_dir))

import polars as pl
import pandas as pd
import pickle
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

from statsmodels.tsa.arima.model import ARIMA
from src.data_prep.prepare_time_splits import (
    get_temporal_splits, 
    prepare_tram_series,
    SELECTED_TRAMS
)


# Paths
DATA_PATH = "data/processed/dataset_imputed_clean.parquet"
MODELS_DIR = "models/arima"
OUTPUT_METRICS = "models/arima/training_metrics.parquet"


def train_arima_model(
    series: pd.Series, 
    order: tuple = (1, 1, 1),
    verbose: bool = True
) -> Dict:
    """
    Entrena un model ARIMA amb els paràmetres especificats.
    
    Parameters
    ----------
    series : pd.Series
        Sèrie temporal amb index datetime
    order : tuple
        Ordre ARIMA (p, d, q)
    verbose : bool
        Si True, mostra informació del procés
    
    Returns
    -------
    dict
        Diccionari amb el model entrenat i mètriques
    """
    try:
        if verbose:
            print(f"Entrenant ARIMA{order}...")
        
        # Entrenem el model
        model = ARIMA(series, order=order)
        fitted_model = model.fit()
        
        # Mètriques
        result = {
            'model': fitted_model,
            'order': order,
            'aic': fitted_model.aic,
            'bic': fitted_model.bic,
            'n_samples': len(series),
            'success': True,
            'error': None
        }
        
        if verbose:
            print(f"AIC: {result['aic']:.2f}, BIC: {result['bic']:.2f}")
        
        return result
        
    except Exception as e:
        if verbose:
            print(f"Error: {e}")
        
        return {
            'model': None,
            'order': order,
            'aic': float('inf'),
            'bic': float('inf'),
            'n_samples': len(series),
            'success': False,
            'error': str(e)
        }


def save_model(model, tram_id: int, order: tuple, models_dir: str = MODELS_DIR) -> str:
    """
    Guarda un model ARIMA entrenat (només paràmetres essencials per reduir espai).
    
    Parameters
    ----------
    model : ARIMAResults
        Model entrenat
    tram_id : int
        ID del tram
    order : tuple
        Ordre ARIMA (p, d, q)
    models_dir : str
        Directori on guardar els models
    
    Returns
    -------
    str
        Path del fitxer guardat
    """
    Path(models_dir).mkdir(parents=True, exist_ok=True)

    # Guardem només informació essencial
    lightweight = {
        "params": model.params,
        "bse": model.bse,
        "aic": model.aic,
        "bic": model.bic,
        "order": order,
    }
    
    model_path = f"{models_dir}/arima_tram_{tram_id}.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(lightweight, f)
    
    return model_path


def load_model(tram_id: int, models_dir: str = MODELS_DIR):
    """
    Carrega un model ARIMA guardat.
    
    Parameters
    ----------
    tram_id : int
        ID del tram
    models_dir : str
        Directori dels models
    
    Returns
    -------
    ARIMAResults
        Model carregat
    """
    model_path = f"{models_dir}/arima_tram_{tram_id}.pkl"
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    return model


def train_all_models(
    tram_ids: List[int] = None,
    order: tuple = (1, 1, 1),
    save_models: bool = True
) -> pl.DataFrame:
    """
    Entrena models ARIMA per a tots els trams utilitzant només dades de TRAIN.
    
    Parameters
    ----------
    tram_ids : list, optional
        Llista d'IDs de trams. Si None, utilitza SELECTED_TRAMS.
    order : tuple
        Ordre ARIMA (p, d, q)
    save_models : bool
        Si True, guarda els models entrenats
    
    Returns
    -------
    pl.DataFrame
        DataFrame amb mètriques d'entrenament
    """
    if tram_ids is None:
        tram_ids = SELECTED_TRAMS

    print(f"Model: ARIMA{order}")
    print(f"Trams: {len(tram_ids)}")
    
    # Carreguem les dades
    df = pl.read_parquet(DATA_PATH)
    
    # Obtenim els splits temporals
    splits_info = get_temporal_splits(df)
    print(f"Train: {splits_info['train_start']} → {splits_info['train_end']}")
    print(f"Val:   {splits_info['val_start']} → {splits_info['val_end']}")
    print(f"Test:  {splits_info['test_start']} → {splits_info['test_end']}")
    print("Utilitzant només dades de TRAIN per entrenar els models")
    
    # Entrenem els models
    results = []
    for i, tram_id in enumerate(tram_ids, 1):
        print(f"[{i}/{len(tram_ids)}] Tram {tram_id}")
        
        # Obtenir dades de TRAIN
        train_data = prepare_tram_series(df, tram_id, 'train', splits_info)
        series = train_data['estatActual']
        
        print(f"Registres de train: {len(series)}")
        print(f"Rang: {series.index.min()} → {series.index.max()}")
        
        # Entrenem el model
        result = train_arima_model(series, order=order, verbose=True)
        
        # Guardem el model si s'ha entrenat correctament
        if result['success'] and save_models:
            model_path = save_model(result['model'], tram_id, order)
            result['model_path'] = model_path
            print(f"Model guardat: {model_path}")
        else:
            result['model_path'] = None
        
        # Afegim informació del tram
        result['idTram'] = tram_id
        result['train_start'] = splits_info['train_start']
        result['train_end'] = splits_info['train_end']
        
        # No guardem el model al DataFrame
        result_without_model = {k: v for k, v in result.items() if k != 'model'}
        results.append(result_without_model)
    
    # Creem un DataFrame amb resultats
    results_df = pl.DataFrame(results)
    
    # Guardem les mètriques
    Path(MODELS_DIR).mkdir(parents=True, exist_ok=True)
    results_df.write_parquet(OUTPUT_METRICS)
    print(f"Mètriques guardades: {OUTPUT_METRICS}")
    
    successful = results_df.filter(pl.col('success') == True).height
    failed = results_df.filter(pl.col('success') == False).height
    
    print(f"Models entrenats correctament: {successful}/{len(tram_ids)}")
    if failed > 0:
        print(f"Models amb errors: {failed}")
        print("\nTrams amb errors:")
        print(results_df.filter(pl.col('success') == False).select(['idTram', 'error']))
    
    # Estadístiques dels models correctes
    if successful > 0:
        successful_df = results_df.filter(pl.col('success') == True)
        
        print(f"Estadístiques (models correctes):")
        print(f"AIC mitjà: {successful_df['aic'].mean():.2f}")
        print(f"AIC min: {successful_df['aic'].min():.2f}")
        print(f"AIC max: {successful_df['aic'].max():.2f}")
        print(f"BIC mitjà: {successful_df['bic'].mean():.2f}")
        print(f"Samples mitjà: {successful_df['n_samples'].mean():.0f}")
    
    print("\nEntrenament completat!")
    
    return results_df


def main():
    """
    Funció principal per entrenar els models baseline.
    """
    # Entrenem tots els models ARIMA(1,1,1)
    results = train_all_models(
        tram_ids=SELECTED_TRAMS,
        order=(1, 1, 1),
        save_models=True
    )
    
    return results


if __name__ == "__main__":
    main()
