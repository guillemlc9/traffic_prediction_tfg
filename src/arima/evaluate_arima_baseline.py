"""
evaluate_arima_baseline.py
---------------------------
Avalua els 30 models ARIMA(1,1,1) sobre els conjunts de validació i test.
Calcula mètriques (MAE, RMSE) i genera visualitzacions.
"""

import sys
from pathlib import Path

# Afegir el directori arrel al PYTHONPATH
root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root_dir))

import polars as pl
import pandas as pd
import numpy as np
import pickle
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import seaborn as sns
import openpyxl
from sklearn.metrics import confusion_matrix

from statsmodels.tsa.arima.model import ARIMA
from src.data_prep.prepare_time_splits import (
    get_temporal_splits, 
    prepare_tram_series,
    SELECTED_TRAMS
)


# Paths
DATA_PATH = "data/processed/dataset_imputed_clean.parquet"
MODELS_DIR = "models/arima"
OUTPUT_METRICS = "models/arima/evaluation_metrics.parquet"
OUTPUT_PLOTS_DIR = "reports/arima"
VARIANCE_CLASSIFICATION_PATH = "data/processed/trams_variancia_classificats.xlsx"


def load_model_params(tram_id: int, models_dir: str = MODELS_DIR) -> Dict:
    """
    Carrega els paràmetres d'un model ARIMA guardat.
    
    Parameters
    ----------
    tram_id : int
        ID del tram
    models_dir : str
        Directori dels models
    
    Returns
    -------
    dict
        Diccionari amb paràmetres del model
    """
    model_path = f"{models_dir}/arima_tram_{tram_id}.pkl"
    
    with open(model_path, 'rb') as f:
        model_params = pickle.load(f)
    
    return model_params


def make_predictions(
    train_series: pd.Series,
    test_series: pd.Series,
    order: tuple = (1, 1, 1)
) -> np.ndarray:
    """
    Entrena un model ARIMA amb les dades de train i fa prediccions sobre test.
    
    Parameters
    ----------
    train_series : pd.Series
        Sèrie temporal d'entrenament
    test_series : pd.Series
        Sèrie temporal de test
    order : tuple
        Ordre ARIMA (p, d, q)
    
    Returns
    -------
    np.ndarray
        Prediccions per al conjunt de test
    """
    # Entrenar model amb dades de train
    model = ARIMA(train_series, order=order)
    fitted_model = model.fit()
    
    # Fer prediccions per al conjunt de test
    n_steps = len(test_series)
    predictions = fitted_model.forecast(steps=n_steps)
    
    return predictions.values


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """
    Calcula mètriques d'error.
    
    Parameters
    ----------
    y_true : np.ndarray
        Valors reals
    y_pred : np.ndarray
        Valors predits
    
    Returns
    -------
    dict
        Diccionari amb mètriques
    """
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
    
    return {
        'mae': mae,
        'rmse': rmse,
        'mape': mape
    }


def evaluate_all_models(
    tram_ids: List[int] = None,
    split: str = 'test'
) -> pl.DataFrame:
    """
    Avalua tots els models sobre el conjunt especificat.
    
    Parameters
    ----------
    tram_ids : list, optional
        Llista d'IDs de trams. Si None, utilitza SELECTED_TRAMS.
    split : str
        Split a avaluar: 'val' o 'test'
    
    Returns
    -------
    pl.DataFrame
        DataFrame amb mètriques per cada tram
    """
    if tram_ids is None:
        tram_ids = SELECTED_TRAMS
    
    print("=" * 60)
    print(f"AVALUACIÓ MODELS ARIMA - Split: {split.upper()}")
    print("=" * 60)
    
    # Carregar dades
    print("\nCarregant dades...")
    df = pl.read_parquet(DATA_PATH)
    
    # Obtenir splits temporals
    splits_info = get_temporal_splits(df)
    print(f"\nAvaluant sobre: {splits_info[f'{split}_start']} → {splits_info[f'{split}_end']}")
    
    # Avaluar models
    results = []
    for i, tram_id in enumerate(tram_ids, 1):
        print(f"\n[{i}/{len(tram_ids)}] Avaluant Tram {tram_id}")
        
        try:
            # Carregar paràmetres del model
            model_params = load_model_params(tram_id)
            order = model_params['order']
            
            # Obtenir dades
            train_data = prepare_tram_series(df, tram_id, 'train', splits_info)
            test_data = prepare_tram_series(df, tram_id, split, splits_info)
            
            train_series = train_data['estatActual']
            test_series = test_data['estatActual']
            
            print(f"  Train: {len(train_series)} registres")
            print(f"  {split.capitalize()}: {len(test_series)} registres")
            
            # Fer prediccions
            predictions = make_predictions(train_series, test_series, order)
            
            # Calcular mètriques
            metrics = calculate_metrics(test_series.values, predictions)
            
            print(f"  MAE: {metrics['mae']:.4f}")
            print(f"  RMSE: {metrics['rmse']:.4f}")
            print(f"  MAPE: {metrics['mape']:.2f}%")
            
            # Guardar resultats
            results.append({
                'idTram': tram_id,
                'split': split,
                'mae': metrics['mae'],
                'rmse': metrics['rmse'],
                'mape': metrics['mape'],
                'n_samples': len(test_series),
                'success': True,
                'error': None
            })
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            results.append({
                'idTram': tram_id,
                'split': split,
                'mae': float('inf'),
                'rmse': float('inf'),
                'mape': float('inf'),
                'n_samples': 0,
                'success': False,
                'error': str(e)
            })
    
    # Crear DataFrame amb resultats
    results_df = pl.DataFrame(results)
    
    # Guardar mètriques
    Path(MODELS_DIR).mkdir(parents=True, exist_ok=True)
    output_path = f"{MODELS_DIR}/evaluation_metrics_{split}.parquet"
    results_df.write_parquet(output_path)
    print(f"\n💾 Mètriques guardades: {output_path}")
    
    # Resum
    print("\n" + "=" * 60)
    print("RESUM AVALUACIÓ")
    print("=" * 60)
    
    successful_df = results_df.filter(pl.col('success') == True)
    
    if successful_df.height > 0:
        print(f"\nModels avaluats correctament: {successful_df.height}/{len(tram_ids)}")
        print(f"\nMètriques mitjanes:")
        print(f"  MAE mitjà: {successful_df['mae'].mean():.4f}")
        print(f"  RMSE mitjà: {successful_df['rmse'].mean():.4f}")
        print(f"  MAPE mitjà: {successful_df['mape'].mean():.2f}%")
        
        print(f"\nDistribució MAE:")
        print(f"  Min: {successful_df['mae'].min():.4f}")
        print(f"  Q1: {successful_df['mae'].quantile(0.25):.4f}")
        print(f"  Mediana: {successful_df['mae'].median():.4f}")
        print(f"  Q3: {successful_df['mae'].quantile(0.75):.4f}")
        print(f"  Max: {successful_df['mae'].max():.4f}")
        
        print(f"\nDistribució RMSE:")
        print(f"  Min: {successful_df['rmse'].min():.4f}")
        print(f"  Q1: {successful_df['rmse'].quantile(0.25):.4f}")
        print(f"  Mediana: {successful_df['rmse'].median():.4f}")
        print(f"  Q3: {successful_df['rmse'].quantile(0.75):.4f}")
        print(f"  Max: {successful_df['rmse'].max():.4f}")
    
    print("\nAvaluació completada!")
    
    return results_df


def create_visualizations(metrics_df: pl.DataFrame, split: str = 'test'):
    """
    Crea visualitzacions de les mètriques.
    
    Parameters
    ----------
    metrics_df : pl.DataFrame
        DataFrame amb mètriques
    split : str
        Split avaluat
    """
    print("\n" + "=" * 60)
    print("GENERANT VISUALITZACIONS")
    print("=" * 60)
    
    # Crear directori de sortida
    Path(OUTPUT_PLOTS_DIR).mkdir(parents=True, exist_ok=True)
    
    # Filtrar només models correctes
    successful_df = metrics_df.filter(pl.col('success') == True).to_pandas()
    
    # Carregar classificació de variància
    variance_df = pl.read_excel(VARIANCE_CLASSIFICATION_PATH)
    variance_dict = dict(zip(
        variance_df['idTram'].to_list(),
        variance_df['categoria_variancia'].to_list()
    ))
    
    # Afegir categoria de variància al DataFrame de mètriques
    successful_df['categoria_variancia'] = successful_df['idTram'].map(variance_dict)
    
    # Configurar estil
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 5)
    
    # Crear figura amb 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. Boxplot de MAE per tram
    ax1 = axes[0]
    successful_df_sorted = successful_df.sort_values('mae')
    sns.boxplot(data=successful_df_sorted, y='mae', ax=ax1, color='skyblue')
    ax1.set_title(f'Distribució MAE per Tram - Split: {split.upper()}', fontsize=14, fontweight='bold')
    ax1.set_ylabel('MAE', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Afegir estadístiques
    mae_mean = successful_df['mae'].mean()
    mae_median = successful_df['mae'].median()
    ax1.axhline(mae_mean, color='red', linestyle='--', linewidth=2, label=f'Mitjana: {mae_mean:.4f}')
    ax1.axhline(mae_median, color='green', linestyle='--', linewidth=2, label=f'Mediana: {mae_median:.4f}')
    ax1.legend()
    
    # 2. Histograma de RMSE
    ax2 = axes[1]
    ax2.hist(successful_df['rmse'], bins=15, color='coral', edgecolor='black', alpha=0.7)
    ax2.set_title(f'Distribució RMSE - Split: {split.upper()}', fontsize=14, fontweight='bold')
    ax2.set_xlabel('RMSE', fontsize=12)
    ax2.set_ylabel('Freqüència', fontsize=12)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Afegir línia de mitjana
    rmse_mean = successful_df['rmse'].mean()
    ax2.axvline(rmse_mean, color='red', linestyle='--', linewidth=2, label=f'Mitjana: {rmse_mean:.4f}')
    ax2.legend()
    
    plt.tight_layout()
    
    # Guardar figura
    output_path = f"{OUTPUT_PLOTS_DIR}/metrics_distribution_{split}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nVisualització guardada: {output_path}")
    
    plt.close()
    
    # Crear gràfic addicional: MAE per tram (barplot amb colors per categoria de variància)
    fig, ax = plt.subplots(figsize=(16, 6))
    
    # Ordenar per MAE
    successful_df_sorted = successful_df.sort_values('mae')
    
    # Definir colors per categoria de variància
    color_map = {
        'alta': '#d62728',      # Vermell
        'mitjana': '#ff7f0e',   # Taronja
        'baixa': '#2ca02c'      # Verd
    }
    
    # Assignar colors segons la categoria de variància
    colors = [color_map.get(cat, 'gray') for cat in successful_df_sorted['categoria_variancia']]
    
    ax.bar(range(len(successful_df_sorted)), successful_df_sorted['mae'], color=colors, alpha=0.7, edgecolor='black')
    ax.set_xticks(range(len(successful_df_sorted)))
    ax.set_xticklabels(successful_df_sorted['idTram'], rotation=45, ha='right')
    ax.set_xlabel('ID Tram', fontsize=12)
    ax.set_ylabel('MAE', fontsize=12)
    ax.set_title(f'MAE per Tram (color per categoria de variància) - Split: {split.upper()}', fontsize=14, fontweight='bold')
    ax.axhline(mae_mean, color='black', linestyle='--', linewidth=2, label=f'Mitjana: {mae_mean:.4f}')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Crear llegenda personalitzada per les categories
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=color_map['alta'], edgecolor='black', alpha=0.7, label='Variància Alta'),
        Patch(facecolor=color_map['mitjana'], edgecolor='black', alpha=0.7, label='Variància Mitjana'),
        Patch(facecolor=color_map['baixa'], edgecolor='black', alpha=0.7, label='Variància Baixa'),
        plt.Line2D([0], [0], color='black', linestyle='--', linewidth=2, label=f'Mitjana MAE: {mae_mean:.4f}')
    ]
    ax.legend(handles=legend_elements, loc='upper left')
    
    plt.tight_layout()
    
    output_path = f"{OUTPUT_PLOTS_DIR}/mae_per_tram_{split}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Visualització guardada: {output_path}")
    
    plt.close()
    
    print("\nVisualitzacions completades!")


def create_confusion_matrix(split: str = 'test'):
    """
    Crea una matriu de confusió per visualitzar la relació entre valors reals (1-5)
    i prediccions arrodonides (1-5).
    
    Aquesta visualització permet veure clarament si els errors es concentren
    a la diagonal (encerts) o a les cel·les veïnes (error de ±1).
    
    Parameters
    ----------
    split : str
        Split a avaluar: 'val' o 'test'
    """
    print("\n" + "=" * 60)
    print(f"GENERANT MATRIU DE CONFUSIÓ - Split: {split.upper()}")
    print("=" * 60)
    
    # Carregar dades
    df = pl.read_parquet(DATA_PATH)
    splits_info = get_temporal_splits(df)
    
    # Llistes per acumular tots els valors reals i prediccions
    all_y_true = []
    all_y_pred = []
    
    # Iterar sobre tots els trams
    for i, tram_id in enumerate(SELECTED_TRAMS, 1):
        print(f"[{i}/{len(SELECTED_TRAMS)}] Processant Tram {tram_id}...", end='\r')
        
        try:
            # Carregar paràmetres del model
            model_params = load_model_params(tram_id)
            order = model_params['order']
            
            # Obtenir dades
            train_data = prepare_tram_series(df, tram_id, 'train', splits_info)
            test_data = prepare_tram_series(df, tram_id, split, splits_info)
            
            train_series = train_data['estatActual']
            test_series = test_data['estatActual']
            
            # Fer prediccions
            predictions = make_predictions(train_series, test_series, order)
            
            # Acumular valors reals i prediccions
            all_y_true.extend(test_series.values)
            all_y_pred.extend(predictions)
            
        except Exception as e:
            print(f"\n  ✗ Error en Tram {tram_id}: {e}")
            continue
    
    print(f"\n\nTotal de prediccions: {len(all_y_true)}")
    
    # Convertir a arrays numpy
    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    
    # Arrodonir prediccions i assegurar que estan en el rang [1, 5]
    all_y_pred_rounded = np.clip(np.round(all_y_pred), 1, 5).astype(int)
    all_y_true_int = np.clip(np.round(all_y_true), 1, 5).astype(int)
    
    # Calcular matriu de confusió
    cm = confusion_matrix(all_y_true_int, all_y_pred_rounded, labels=[1, 2, 3, 4, 5])
    
    # Calcular percentatges per fila (normalitzar per valors reals)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Crear figura amb dues matrius: valors absoluts i percentatges
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    # 1. Matriu de confusió amb valors absoluts
    ax1 = axes[0]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=[1, 2, 3, 4, 5], 
                yticklabels=[1, 2, 3, 4, 5],
                cbar_kws={'label': 'Nombre de prediccions'},
                ax=ax1, linewidths=0.5, linecolor='gray')
    ax1.set_xlabel('Predicció Arrodonida', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Valor Real (estatActual)', fontsize=12, fontweight='bold')
    ax1.set_title(f'Matriu de Confusió - Valors Absoluts\nSplit: {split.upper()}', 
                  fontsize=14, fontweight='bold')
    
    # 2. Matriu de confusió amb percentatges
    ax2 = axes[1]
    sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Greens', 
                xticklabels=[1, 2, 3, 4, 5], 
                yticklabels=[1, 2, 3, 4, 5],
                cbar_kws={'label': 'Percentatge (%)'},
                ax=ax2, linewidths=0.5, linecolor='gray')
    ax2.set_xlabel('Predicció Arrodonida', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Valor Real (estatActual)', fontsize=12, fontweight='bold')
    ax2.set_title(f'Matriu de Confusió - Percentatges per Fila\nSplit: {split.upper()}', 
                  fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Guardar figura
    output_path = f"{OUTPUT_PLOTS_DIR}/confusion_matrix_{split}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Matriu de confusió guardada: {output_path}")
    
    plt.close()
    
    # Calcular i mostrar estadístiques sobre la diagonal
    diagonal_sum = np.trace(cm)
    near_diagonal_sum = 0
    for i in range(5):
        for j in range(5):
            if abs(i - j) == 1:  # Cel·les veïnes
                near_diagonal_sum += cm[i, j]
    
    total_predictions = cm.sum()
    diagonal_percent = (diagonal_sum / total_predictions) * 100
    near_diagonal_percent = (near_diagonal_sum / total_predictions) * 100
    far_errors_percent = 100 - diagonal_percent - near_diagonal_percent
    
    print("\n" + "=" * 60)
    print("ANÀLISI DE LA MATRIU DE CONFUSIÓ")
    print("=" * 60)
    print(f"\nPrediccions exactes (diagonal): {diagonal_sum} ({diagonal_percent:.2f}%)")
    print(f"Errors de ±1 (cel·les veïnes): {near_diagonal_sum} ({near_diagonal_percent:.2f}%)")
    print(f"Errors llunyans (±2 o més): {total_predictions - diagonal_sum - near_diagonal_sum} ({far_errors_percent:.2f}%)")
    print(f"\nTotal prediccions: {total_predictions}")
    
    # Calcular MAE per validar
    mae = np.mean(np.abs(all_y_true - all_y_pred))
    print(f"\nMAE global: {mae:.4f}")
    
    print("\nMatriu de confusió completada!")


def main():
    """
    Funció principal per avaluar els models baseline.
    """
    # Avaluar sobre el conjunt de test
    print("Avaluant models sobre el conjunt de TEST...")
    metrics_test = evaluate_all_models(
        tram_ids=SELECTED_TRAMS,
        split='test'
    )
    
    # Crear visualitzacions
    create_visualizations(metrics_test, split='test')
    
    # Crear matriu de confusió per TEST
    create_confusion_matrix(split='test')
    
    # Avaluem sobre Val
    print("\n\n" + "=" * 60)
    print("Avaluant models sobre el conjunt de VALIDACIÓ...")
    metrics_val = evaluate_all_models(
        tram_ids=SELECTED_TRAMS,
        split='val'
    )
    
    create_visualizations(metrics_val, split='val')
    
    # Crear matriu de confusió per VAL
    create_confusion_matrix(split='val')
    
    return metrics_test, metrics_val


if __name__ == "__main__":
    main()

