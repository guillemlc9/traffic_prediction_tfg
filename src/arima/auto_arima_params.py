"""
auto_arima_params.py
--------------------
Utilitza Auto ARIMA per trobar els millors paràmetres (p,d,q) automàticament.
Compara els resultats amb el mètode heurístic de determine_params.py
"""

import polars as pl
import pandas as pd
from pmdarima import auto_arima
import warnings
warnings.filterwarnings('ignore')

# Paths
DATA_PATH = "data/processed/dataset_imputed_clean.parquet"
OUTPUT_PATH = "data/processed/auto_arima_params.parquet"


def find_best_arima(series: pd.Series, seasonal: bool = False, verbose: bool = True) -> dict:
    """
    Utilitza Auto ARIMA per trobar els millors paràmetres.
    
    Parameters
    ----------
    series : pd.Series
        Sèrie temporal a analitzar
    seasonal : bool
        Si True, també busca paràmetres estacionals
    verbose : bool
        Si True, mostra informació del procés
    
    Returns
    -------
    dict amb els resultats
    """
    try:
        if verbose:
            print("  Executant Auto ARIMA...")
        
        model = auto_arima(
            series.dropna(),
            start_p=0, max_p=5,
            start_q=0, max_q=5,
            d=None,  # Determina automàticament
            max_d=2,
            seasonal=seasonal,
            m=288 if seasonal else 1,  # 288 = 1 dia (intervals de 5 min)
            start_P=0, max_P=2,
            start_Q=0, max_Q=2,
            D=None,
            trace=False,
            error_action='ignore',
            suppress_warnings=True,
            stepwise=True,  # Més ràpid
            n_jobs=-1,
            information_criterion='aic'
        )
        
        result = {
            'p': model.order[0],
            'd': model.order[1],
            'q': model.order[2],
            'aic': model.aic(),
            'bic': model.bic(),
            'arima_order': f"({model.order[0]},{model.order[1]},{model.order[2]})"
        }
        
        if seasonal:
            P, D, Q, m = model.seasonal_order
            result['P'] = P
            result['D'] = D
            result['Q'] = Q
            result['m'] = m
            result['sarima_order'] = f"({model.order[0]},{model.order[1]},{model.order[2]})x({P},{D},{Q},{m})"
        
        return result
        
    except Exception as e:
        print(f"Error: {e}")
        return {
            'p': 0,
            'd': 0,
            'q': 0,
            'aic': float('inf'),
            'bic': float('inf'),
            'arima_order': '(0,0,0)',
            'error': str(e)
        }


def main(trams: list = None, seasonal: bool = False):
    """
    Analitza els trams amb Auto ARIMA.
    
    Parameters
    ----------
    trams : list
        Llista d'IDs de trams a analitzar
    seasonal : bool
        Si True, busca també paràmetres estacionals (més lent)
    """
    print("=" * 60)
    print("AUTO ARIMA - Cerca Automàtica de Paràmetres")
    print("=" * 60)
    
    # Llegim les dades
    df = pl.read_parquet(DATA_PATH)
    
    if trams is None:
        print("ERROR: Cal especificar els trams a analitzar")
        return
    
    print(f"Trams a analitzar: {len(trams)}")
    if seasonal:
        print("⚠️  Mode estacional activat (pot trigar més)")
    
    # Analitzem cada tram
    results = []
    for i, tram_id in enumerate(trams, 1):
        print(f"\n[{i}/{len(trams)}] Tram {tram_id}")
        
        # Filtrem les dades
        df_tram = (
            df.filter(pl.col("idTram") == tram_id)
            .select(["timestamp", "estatActual"])
            .sort("timestamp")
            .to_pandas()
        )
        
        series = df_tram.set_index('timestamp')['estatActual']
        
        # Auto ARIMA
        result = find_best_arima(series, seasonal=seasonal, verbose=True)
        result['idTram'] = tram_id
        results.append(result)
        
        # Mostrem resultat
        if seasonal and 'sarima_order' in result:
            print(f"  → Millor model: SARIMA{result['sarima_order']}")
        else:
            print(f"  → Millor model: ARIMA{result['arima_order']}")
        print(f"  → AIC: {result['aic']:.2f}")
        print(f"  → BIC: {result['bic']:.2f}")
    
    # Guardem resultats
    results_df = pl.DataFrame(results)
    results_df.write_parquet(OUTPUT_PATH)
    print(f"Resultats guardats a: {OUTPUT_PATH}")
    
    # Resum
    print("=" * 60)
    print("RESUM")
    print("=" * 60)
    
    if seasonal:
        print(results_df.select([
            "idTram", "p", "d", "q", "P", "D", "Q", "m", "aic", "sarima_order"
        ]))
    else:
        print(results_df.select([
            "idTram", "p", "d", "q", "aic", "bic", "arima_order"
        ]))
    
    # Estadístiques
    print("\nDistribució de paràmetres:")
    print(f"d=0: {(results_df['d'] == 0).sum()} trams")
    print(f"d=1: {(results_df['d'] == 1).sum()} trams")
    print(f"d=2: {(results_df['d'] == 2).sum()} trams")
    
    print(f"\nRang de p: {results_df['p'].min()} - {results_df['p'].max()}")
    print(f"Rang de q: {results_df['q'].min()} - {results_df['q'].max()}")
    
    print(f"\nAIC mitjà: {results_df['aic'].mean():.2f}")
    print(f"BIC mitjà: {results_df['bic'].mean():.2f}")
    
    return results_df


if __name__ == "__main__":
    # Els teus 30 trams
    trams = [233,158,57,526,83,445,11,22,460,178,126,534,164,50,278,388,478,270,332,
             254,224,293,117,100,104,409,31,221,360,303]

    # Executem Auto ARIMA (sense estacionalitat per ara)
    results = main(trams=trams, seasonal=False)
