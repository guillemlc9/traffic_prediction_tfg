"""
compare_models_by_variance.py
------------------------------
Compara les mètriques dels models ARIMA i LSTM agrupades per categoria de variància.

Aquest script:
1. Carrega les mètriques d'ARIMA (per tram)
2. Carrega les mètriques d'LSTM (per tram)
3. Carrega la classificació de variància dels trams
4. Agrupa les mètriques per categoria de variància (alta, mitjana, baixa)
5. Genera una taula comparativa i visualitzacions
"""

import sys
from pathlib import Path

# Afegir el directori arrel al PYTHONPATH
root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root_dir))

import polars as pl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict


def load_arima_metrics(metrics_path: str = "models/arima/evaluation_metrics_test.parquet") -> pl.DataFrame:
    """
    Carrega les mètriques d'ARIMA.
    
    Returns
    -------
    pl.DataFrame
        DataFrame amb columnes: idTram, mae, rmse, ...
    """
    df = pl.read_parquet(metrics_path)
    # Filtrar només els models correctes
    df = df.filter(pl.col('success') == True)
    return df.select(['idTram', 'mae', 'rmse', 'n_samples'])


def load_lstm_metrics(
    model_dir: str = "models/lstm/20251227_143405_lr1e-03_bs1024_u64_w36_h1"
) -> pl.DataFrame:
    """
    Carrega les mètriques d'LSTM per tram.
    
    Returns
    -------
    pl.DataFrame
        DataFrame amb columnes: idTram, mae, rmse, ...
    """
    metrics_path = Path(model_dir) / "metrics_per_tram.parquet"
    df = pl.read_parquet(metrics_path)
    return df.select(['idTram', 'mae', 'rmse', 'n_samples'])


def load_variance_classification(
    variance_path: str = "data/processed/trams_variancia_classificats.xlsx"
) -> pl.DataFrame:
    """
    Carrega la classificació de variància dels trams.
    
    Returns
    -------
    pl.DataFrame
        DataFrame amb columnes: idTram, categoria_variancia
    """
    df = pl.read_excel(variance_path)
    return df.select(['idTram', 'categoria_variancia'])


def aggregate_by_variance(
    arima_metrics: pl.DataFrame,
    lstm_metrics: pl.DataFrame,
    variance_classification: pl.DataFrame
) -> pl.DataFrame:
    """
    Agrupa les mètriques per categoria de variància.
    
    Returns
    -------
    pl.DataFrame
        DataFrame amb columnes: categoria_variancia, arima_mae, arima_rmse, 
                                lstm_mae, lstm_rmse, n_trams
    """
    # Afegir categoria de variància a les mètriques
    arima_with_var = arima_metrics.join(variance_classification, on='idTram', how='left')
    lstm_with_var = lstm_metrics.join(variance_classification, on='idTram', how='left')
    
    # Agrupar per categoria de variància
    arima_agg = (
        arima_with_var
        .group_by('categoria_variancia')
        .agg([
            pl.col('mae').mean().alias('arima_mae'),
            pl.col('rmse').mean().alias('arima_rmse'),
            pl.col('idTram').count().alias('n_trams')
        ])
    )
    
    lstm_agg = (
        lstm_with_var
        .group_by('categoria_variancia')
        .agg([
            pl.col('mae').mean().alias('lstm_mae'),
            pl.col('rmse').mean().alias('lstm_rmse')
        ])
    )
    
    # Combinar resultats
    comparison = arima_agg.join(lstm_agg, on='categoria_variancia', how='left')
    
    # Ordenar per categoria (alta, mitjana, baixa)
    category_order = {'alta': 0, 'mitjana': 1, 'baixa': 2}
    comparison = comparison.with_columns(
        pl.col('categoria_variancia').map_elements(
            lambda x: category_order.get(x, 999),
            return_dtype=pl.Int64
        ).alias('order')
    ).sort('order').drop('order')
    
    return comparison


def create_comparison_table(comparison_df: pl.DataFrame) -> pd.DataFrame:
    """
    Crea una taula comparativa formatada.
    
    Returns
    -------
    pd.DataFrame
        Taula comparativa amb format llegible
    """
    # Convertir a pandas per millor formatació
    df = comparison_df.to_pandas()
    
    # Renombrar columnes per claredat
    df = df.rename(columns={
        'categoria_variancia': 'Categoria Variància',
        'arima_mae': 'ARIMA MAE',
        'arima_rmse': 'ARIMA RMSE',
        'lstm_mae': 'LSTM MAE',
        'lstm_rmse': 'LSTM RMSE',
        'n_trams': 'Nombre de Trams'
    })
    
    # Calcular diferències
    df['Δ MAE (LSTM - ARIMA)'] = df['LSTM MAE'] - df['ARIMA MAE']
    df['Δ RMSE (LSTM - ARIMA)'] = df['LSTM RMSE'] - df['ARIMA RMSE']
    
    # Calcular percentatges de millora
    df['% Millora MAE'] = ((df['ARIMA MAE'] - df['LSTM MAE']) / df['ARIMA MAE']) * 100
    df['% Millora RMSE'] = ((df['ARIMA RMSE'] - df['LSTM RMSE']) / df['ARIMA RMSE']) * 100
    
    return df


def create_visualizations(comparison_df: pl.DataFrame, output_dir: str = "reports/comparison"):
    """
    Crea visualitzacions comparatives.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Convertir a pandas per matplotlib
    df = comparison_df.to_pandas()
    
    # Configurar estil
    sns.set_style("whitegrid")
    
    # 1. Gràfic de barres comparant MAE
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # MAE
    ax1 = axes[0]
    x = np.arange(len(df))
    width = 0.35
    
    ax1.bar(x - width/2, df['arima_mae'], width, label='ARIMA', color='#1f77b4', alpha=0.8)
    ax1.bar(x + width/2, df['lstm_mae'], width, label='LSTM', color='#ff7f0e', alpha=0.8)
    
    ax1.set_xlabel('Categoria de Variància', fontsize=12, fontweight='bold')
    ax1.set_ylabel('MAE', fontsize=12, fontweight='bold')
    ax1.set_title('Comparació MAE: ARIMA vs LSTM', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(df['categoria_variancia'].str.capitalize())
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # RMSE
    ax2 = axes[1]
    ax2.bar(x - width/2, df['arima_rmse'], width, label='ARIMA', color='#1f77b4', alpha=0.8)
    ax2.bar(x + width/2, df['lstm_rmse'], width, label='LSTM', color='#ff7f0e', alpha=0.8)
    
    ax2.set_xlabel('Categoria de Variància', fontsize=12, fontweight='bold')
    ax2.set_ylabel('RMSE', fontsize=12, fontweight='bold')
    ax2.set_title('Comparació RMSE: ARIMA vs LSTM', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(df['categoria_variancia'].str.capitalize())
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/comparison_by_variance.png", dpi=300, bbox_inches='tight')
    print(f"✓ Visualització guardada: {output_dir}/comparison_by_variance.png")
    plt.close()
    
    # 2. Gràfic de millora percentual
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calcular millora percentual
    mae_improvement = ((df['arima_mae'] - df['lstm_mae']) / df['arima_mae']) * 100
    rmse_improvement = ((df['arima_rmse'] - df['lstm_rmse']) / df['arima_rmse']) * 100
    
    x = np.arange(len(df))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, mae_improvement, width, label='MAE', alpha=0.8)
    bars2 = ax.bar(x + width/2, rmse_improvement, width, label='RMSE', alpha=0.8)
    
    # Colorar barres segons si hi ha millora o empitjorament
    for bars in [bars1, bars2]:
        for bar in bars:
            if bar.get_height() < 0:
                bar.set_color('#d62728')  # Vermell per empitjorament
            else:
                bar.set_color('#2ca02c')  # Verd per millora
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax.set_xlabel('Categoria de Variància', fontsize=12, fontweight='bold')
    ax.set_ylabel('% Millora (positiu = LSTM millor)', fontsize=12, fontweight='bold')
    ax.set_title('Millora Percentual: LSTM vs ARIMA', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(df['categoria_variancia'].str.capitalize())
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/improvement_by_variance.png", dpi=300, bbox_inches='tight')
    print(f"✓ Visualització guardada: {output_dir}/improvement_by_variance.png")
    plt.close()


def print_summary(comparison_df: pl.DataFrame, table_df: pd.DataFrame):
    """
    Mostra un resum de la comparació.
    """
    print("\n" + "=" * 80)
    print("COMPARACIÓ ARIMA vs LSTM PER CATEGORIA DE VARIÀNCIA")
    print("=" * 80)
    
    print("\nTaula Comparativa:")
    print("-" * 80)
    
    # Formatació de la taula
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.float_format', lambda x: f'{x:.4f}')
    
    print(table_df.to_string(index=False))
    
    print("\n" + "=" * 80)
    print("RESUM GLOBAL")
    print("=" * 80)
    
    # Mitjanes globals
    arima_mae_global = comparison_df['arima_mae'].mean()
    arima_rmse_global = comparison_df['arima_rmse'].mean()
    lstm_mae_global = comparison_df['lstm_mae'].mean()
    lstm_rmse_global = comparison_df['lstm_rmse'].mean()
    
    print(f"\nMètriques Globals (mitjana de categories):")
    print(f"  ARIMA - MAE:  {arima_mae_global:.4f}")
    print(f"  ARIMA - RMSE: {arima_rmse_global:.4f}")
    print(f"  LSTM  - MAE:  {lstm_mae_global:.4f}")
    print(f"  LSTM  - RMSE: {lstm_rmse_global:.4f}")
    
    mae_diff = lstm_mae_global - arima_mae_global
    rmse_diff = lstm_rmse_global - arima_rmse_global
    
    print(f"\nDiferències (LSTM - ARIMA):")
    print(f"  Δ MAE:  {mae_diff:+.4f} ({(mae_diff/arima_mae_global)*100:+.2f}%)")
    print(f"  Δ RMSE: {rmse_diff:+.4f} ({(rmse_diff/arima_rmse_global)*100:+.2f}%)")
    
    # Determinar quin model és millor
    if mae_diff < 0 and rmse_diff < 0:
        print("\n✅ LSTM és millor que ARIMA en ambdues mètriques")
    elif mae_diff > 0 and rmse_diff > 0:
        print("\n✅ ARIMA és millor que LSTM en ambdues mètriques")
    else:
        print("\n⚖️  Els models tenen rendiments mixtos segons la mètrica")
    
    print("=" * 80)


def main():
    """
    Script principal per comparar models.
    """
    OUTPUT_DIR = "reports/comparison"
    
    print("=" * 80)
    print("COMPARACIÓ DE MODELS: ARIMA vs LSTM")
    print("=" * 80)
    
    # 1. Carregar mètriques
    print("\n1. Carregant mètriques...")
    arima_metrics = load_arima_metrics()
    print(f"   ✓ ARIMA: {arima_metrics.height} trams")
    
    lstm_metrics = load_lstm_metrics()
    print(f"   ✓ LSTM: {lstm_metrics.height} trams")
    
    variance_classification = load_variance_classification()
    print(f"   ✓ Classificació de variància: {variance_classification.height} trams")
    
    # 2. Agregar per variància
    print("\n2. Agregant mètriques per categoria de variància...")
    comparison_df = aggregate_by_variance(arima_metrics, lstm_metrics, variance_classification)
    
    # 3. Crear taula comparativa
    print("\n3. Creant taula comparativa...")
    table_df = create_comparison_table(comparison_df)
    
    # 4. Guardar resultats
    print("\n4. Guardant resultats...")
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    # Guardar com Excel
    excel_path = f"{OUTPUT_DIR}/model_comparison_by_variance.xlsx"
    table_df.to_excel(excel_path, index=False, engine='openpyxl')
    print(f"   ✓ Taula Excel guardada: {excel_path}")
    
    # Guardar com CSV
    csv_path = f"{OUTPUT_DIR}/model_comparison_by_variance.csv"
    table_df.to_csv(csv_path, index=False)
    print(f"   ✓ Taula CSV guardada: {csv_path}")
    
    # Guardar com Parquet
    parquet_path = f"{OUTPUT_DIR}/model_comparison_by_variance.parquet"
    comparison_df.write_parquet(parquet_path)
    print(f"   ✓ Dades Parquet guardades: {parquet_path}")
    
    # 5. Crear visualitzacions
    print("\n5. Creant visualitzacions...")
    create_visualizations(comparison_df, OUTPUT_DIR)
    
    # 6. Mostrar resum
    print_summary(comparison_df, table_df)
    
    print("\n" + "=" * 80)
    print("✅ COMPARACIÓ COMPLETADA!")
    print(f"   Resultats guardats a: {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
