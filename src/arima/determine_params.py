"""
determine_params.py
-------------------
Mòdul per determinar els paràmetres òptims (p, d, q) per models ARIMA
per a cada tram de trànsit seleccionat.

Anàlisi que es fa:
1. Test de estacionarietat (ADF test) per determinar 'd'
2. ACF (Autocorrelation Function) per determinar 'q'
3. PACF (Partial Autocorrelation Function) per determinar 'p'
4. Auto ARIMA per trobar els millors paràmetres automàticament
"""

import polars as pl
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pmdarima import auto_arima
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')

# Paths
DATA_PATH = "data/processed/dataset_imputed_clean.parquet"
OUTPUT_PATH = "data/processed/arima_params_analysis.parquet"


def test_stationarity(series: pd.Series, alpha: float = 0.05) -> Dict:
    """
    Test de Dickey-Fuller Augmentat per comprovar estacionarietat.
    
    Returns
    -------
    dict amb:
        - is_stationary: bool
        - adf_statistic: float
        - p_value: float
        - n_lags_used: int
        - critical_values: dict
        - d_suggested: int (0 si estacionària, 1 si no)
    """
    result = adfuller(series.dropna(), autolag='AIC')
    
    is_stationary = result[1] < alpha
    
    return {
        'is_stationary': is_stationary,
        'adf_statistic': result[0],
        'p_value': result[1],
        'n_lags_used': result[2],
        'critical_values': result[4],
        'd_suggested': 0 if is_stationary else 1
    }


def determine_p_from_pacf(series: pd.Series, max_lags: int = 40, threshold: float = 0.05) -> int:
    """
    Determina 'p' analitzant la PACF.
    
    'p' és l'ordre del component AR (autoregressive).
    Es determina pel nombre de lags significatius a la PACF.
    """
    pacf_values = pacf(series.dropna(), nlags=max_lags, method='ywm')
    
    # Calculem el llindar de significància (aproximat)
    n = len(series.dropna())
    significance_level = 1.96 / np.sqrt(n)
    
    # Comptem quants lags són significatius
    significant_lags = []
    for i in range(1, len(pacf_values)):
        if abs(pacf_values[i]) > significance_level:
            significant_lags.append(i)
    
    # 'p' és típicament el primer lag significatiu o la mitjana dels primers
    if len(significant_lags) == 0:
        return 0
    elif len(significant_lags) <= 3:
        return max(significant_lags)
    else:
        # Si hi ha molts lags significatius, agafem els primers 3
        return max(significant_lags[:3])


def determine_q_from_acf(series: pd.Series, max_lags: int = 40, threshold: float = 0.05) -> int:
    """
    Determina 'q' analitzant la ACF.
    
    'q' és l'ordre del component MA (moving average).
    Es determina pel nombre de lags significatius a la ACF.
    """
    acf_values = acf(series.dropna(), nlags=max_lags, fft=False)
    
    # Calculem el llindar de significància
    n = len(series.dropna())
    significance_level = 1.96 / np.sqrt(n)
    
    # Comptem quants lags són significatius
    significant_lags = []
    for i in range(1, len(acf_values)):
        if abs(acf_values[i]) > significance_level:
            significant_lags.append(i)
    
    # 'q' és típicament el primer lag significatiu o la mitjana dels primers
    if len(significant_lags) == 0:
        return 0
    elif len(significant_lags) <= 3:
        return max(significant_lags)
    else:
        return max(significant_lags[:3])


def analyze_tram(df_tram: pd.DataFrame, tram_id: int) -> Dict:
    """
    Analitza un tram i determina els paràmetres ARIMA suggerits.
    
    Parameters
    ----------
    df_tram : pd.DataFrame
        Dades del tram amb columnes ['timestamp', 'estatActual']
    tram_id : int
        ID del tram
    
    Returns
    -------
    dict amb els resultats de l'anàlisi
    """
    # Preparem la sèrie temporal
    series = df_tram.set_index('timestamp')['estatActual']
    
    # 1. Test d'estacionarietat
    stationarity = test_stationarity(series)
    d = stationarity['d_suggested']
    
    # 2. Si no és estacionària, la diferenciem
    if d == 1:
        series_diff = series.diff().dropna()
        # Tornem a testejar
        stationarity_diff = test_stationarity(series_diff)
        if not stationarity_diff['is_stationary']:
            d = 2  # Necessitem segona diferenciació
            series_for_analysis = series.diff().diff().dropna()
        else:
            series_for_analysis = series_diff
    else:
        series_for_analysis = series
    
    # 3. Determinem p i q
    p = determine_p_from_pacf(series_for_analysis)
    q = determine_q_from_acf(series_for_analysis)
    
    # 4. Estadístiques bàsiques
    n_obs = len(series)
    n_nulls = series.isna().sum()
    mean_val = series.mean()
    std_val = series.std()
    
    return {
        'idTram': tram_id,
        'n_observations': n_obs,
        'n_nulls': n_nulls,
        'mean': mean_val,
        'std': std_val,
        'is_stationary': stationarity['is_stationary'],
        'adf_statistic': stationarity['adf_statistic'],
        'adf_pvalue': stationarity['p_value'],
        'p_suggested': p,
        'd_suggested': d,
        'q_suggested': q,
        'arima_order': f"({p},{d},{q})"
    }


def plot_diagnostics(df_tram: pd.DataFrame, tram_id: int, save_path: str = None):
    """
    Crea gràfics de diagnòstic per un tram:
    - Sèrie temporal original
    - ACF
    - PACF
    - Sèrie diferenciada (si cal)
    """
    series = df_tram.set_index('timestamp')['estatActual']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Anàlisi ARIMA - Tram {tram_id}', fontsize=16)
    
    # 1. Sèrie temporal original
    axes[0, 0].plot(series.index, series.values)
    axes[0, 0].set_title('Sèrie Temporal Original')
    axes[0, 0].set_xlabel('Timestamp')
    axes[0, 0].set_ylabel('Estat Actual')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. ACF
    plot_acf(series.dropna(), lags=40, ax=axes[0, 1])
    axes[0, 1].set_title('Autocorrelation Function (ACF)')
    
    # 3. PACF
    plot_pacf(series.dropna(), lags=40, ax=axes[1, 0], method='ywm')
    axes[1, 0].set_title('Partial Autocorrelation Function (PACF)')
    
    # 4. Sèrie diferenciada
    series_diff = series.diff().dropna()
    axes[1, 1].plot(series_diff.index, series_diff.values)
    axes[1, 1].set_title('Sèrie Diferenciada (d=1)')
    axes[1, 1].set_xlabel('Timestamp')
    axes[1, 1].set_ylabel('Diferència')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Gràfic guardat a: {save_path}")
    
    plt.close()


def main(trams_to_analyze: List[int] = None, create_plots: bool = False):
    """
    Analitza els trams seleccionats i determina els paràmetres ARIMA.
    
    Parameters
    ----------
    trams_to_analyze : List[int], optional
        Llista d'IDs de trams a analitzar. Si None, analitza els 30 primers
        amb més variància i menys imputació.
    create_plots : bool
        Si True, crea gràfics de diagnòstic per cada tram
    """
    # Llegim les dades
    print("Carregant dades...")
    df = pl.read_parquet(DATA_PATH)
    
    # Si no s'especifiquen trams, seleccionem els millors
    if trams_to_analyze is None:
        print("\nSeleccionant els 30 millors trams...")
        # Calculem variància i % imputació per tram
        tram_stats = df.group_by("idTram").agg([
            pl.col("estatActual").var().alias("variancia"),
            (pl.col("is_gap").sum() / pl.len() * 100).alias("pct_gaps")
        ])
        
        # Seleccionem trams amb alta variància i baixa imputació
        trams_seleccionats = (
            tram_stats
            .filter(pl.col("pct_gaps") < 75)  # Menys del 75% de gaps
            .sort("variancia", descending=True)
            .head(30)
        )
        
        trams_to_analyze = trams_seleccionats["idTram"].to_list()
        print(f"Trams seleccionats: {trams_to_analyze}")
    
    # Analitzem cada tram
    results = []
    for i, tram_id in enumerate(trams_to_analyze, 1):
        print(f"\n[{i}/{len(trams_to_analyze)}] Analitzant tram {tram_id}...")
        
        # Filtrem les dades del tram
        df_tram = (
            df.filter(pl.col("idTram") == tram_id)
            .select(["timestamp", "estatActual"])
            .sort("timestamp")
            .to_pandas()
        )
        
        # Analitzem
        result = analyze_tram(df_tram, tram_id)
        results.append(result)
        
        print(f"  → Paràmetres suggerits: ARIMA{result['arima_order']}")
        print(f"  → Estacionària: {result['is_stationary']}")
        print(f"  → ADF p-value: {result['adf_pvalue']:.4f}")
        
        # Crear gràfics si cal
        if create_plots:
            plot_path = f"reports/arima_diagnostics_tram_{tram_id}.png"
            plot_diagnostics(df_tram, tram_id, save_path=plot_path)
    
    # Guardem els resultats
    results_df = pl.DataFrame(results)
    results_df.write_parquet(OUTPUT_PATH)
    print(f"\n✅ Resultats guardats a: {OUTPUT_PATH}")
    
    # Resum
    print("\n" + "=" * 60)
    print("RESUM DE L'ANÀLISI")
    print("=" * 60)
    print(results_df.select([
        "idTram", "n_observations", "is_stationary", 
        "p_suggested", "d_suggested", "q_suggested", "arima_order"
    ]))
    
    # Estadístiques dels paràmetres
    print("\nDistribució de paràmetres:")
    print(f"d=0: {(results_df['d_suggested'] == 0).sum()} trams")
    print(f"d=1: {(results_df['d_suggested'] == 1).sum()} trams")
    print(f"d=2: {(results_df['d_suggested'] == 2).sum()} trams")
    
    print(f"\nRang de p: {results_df['p_suggested'].min()} - {results_df['p_suggested'].max()}")
    print(f"Rang de q: {results_df['q_suggested'].min()} - {results_df['q_suggested'].max()}")
    
    return results_df


if __name__ == "__main__":
    # Pots especificar els trams aquí o deixar que es seleccionin automàticament
    trams = [233,158,57,526,83,445,11,22,460,178,126,534,164,50,278,293,388,478,
            270,332,254,224,117,100,104,409,31,221,360,303]
    
    results = main(trams_to_analyze=trams, create_plots=False)
