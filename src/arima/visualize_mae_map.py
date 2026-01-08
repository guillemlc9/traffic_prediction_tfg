"""
visualize_mae_map.py
--------------------
Crea un mapa de Barcelona mostrant els trams amb colors segons el seu MAE.
Colors: verd (MAE baix), taronja (MAE mitjà), vermell (MAE alt)
"""

import sys
from pathlib import Path

# Afegim el directori arrel al PYTHONPATH
root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root_dir))

import polars as pl
import pandas as pd
import folium
from folium import plugins
import warnings
warnings.filterwarnings('ignore')


# Paths
METRICS_PATH = "models/arima/evaluation_metrics_test.parquet"
TRAMS_PATH = "data/transit_relacio_trams_format_long.csv"
OUTPUT_MAP = "reports/arima/mae_map.html"


def get_color_from_mae(mae: float, mae_mean: float) -> str:
    """
    Retorna el color segons el valor de MAE.
    
    Parameters
    ----------
    mae : float
        Valor de MAE
    mae_mean : float
        MAE mitjà
    
    Returns
    -------
    str
        Color en format hex
    """
    # Mateixos criteris que el gràfic de barres
    if mae < mae_mean:
        return '#2ecc71'  # Verd
    elif mae < mae_mean * 1.5:
        return '#f39c12'  # Taronja
    else:
        return '#e74c3c'  # Vermell


def create_mae_map():
    """
    Crea un mapa amb els trams colorejats segons el seu MAE.
    """
    
    # Carreguem les mètriques
    print("\nCarregant mètriques...")
    metrics = pl.read_parquet(METRICS_PATH)
    metrics_ok = metrics.filter(pl.col('success') == True)
    mae_mean = metrics_ok['mae'].mean()
    
    print(f"MAE mitjà: {mae_mean:.4f}")
    
    # Carreguem les dades de trams
    print("Carregant dades de trams...")
    trams_df = pl.read_csv(TRAMS_PATH)
    
    # Creem el mapa centrat a Barcelona amb estil gris clar
    barcelona_center = [41.3851, 2.1734]
    m = folium.Map(
        location=barcelona_center,
        zoom_start=13,
        tiles='CartoDB positron'
    )
    
    # Afegim els trams al mapa
    print("\nAfegint trams al mapa...")
    trams_added = 0
    
    for row in metrics_ok.iter_rows(named=True):
        tram_id = row['idTram']
        mae = row['mae']
        
        # Obtenim les coordenades del tram
        tram_coords = trams_df.filter(pl.col('Tram') == tram_id)
        
        if tram_coords.height == 0:
            print(f"Tram {tram_id}: No s'han trobat coordenades")
            continue
        
        # Convertim a pandas per facilitar l'ús amb folium
        tram_coords_pd = tram_coords.to_pandas()
        
        # Creem la llista de coordenades (lat, lon)
        coordinates = list(zip(
            tram_coords_pd['Latitud'],
            tram_coords_pd['Longitud']
        ))
        
        # Determinem el color
        color = get_color_from_mae(mae, mae_mean)
        
        # Obtenim la descripció del tram
        descripcio = tram_coords_pd['Descripció'].iloc[0] if len(tram_coords_pd) > 0 else f"Tram {tram_id}"
        
        # Creem el popup amb informació
        popup_html = f"""
        <div style="font-family: Arial; width: 200px;">
            <h4 style="margin: 0; color: {color};">Tram {tram_id}</h4>
            <p style="margin: 5px 0;"><b>{descripcio}</b></p>
            <hr style="margin: 5px 0;">
            <p style="margin: 3px 0;"><b>MAE:</b> {mae:.4f}</p>
            <p style="margin: 3px 0;"><b>RMSE:</b> {row['rmse']:.4f}</p>
            <p style="margin: 3px 0;"><b>MAPE:</b> {row['mape']:.2f}%</p>
        </div>
        """
        
        # Afegim la línia al mapa
        folium.PolyLine(
            coordinates,
            color=color,
            weight=5,
            opacity=0.8,
            popup=folium.Popup(popup_html, max_width=250),
            tooltip=f"Tram {tram_id}: MAE={mae:.4f}"
        ).add_to(m)
        
        trams_added += 1
        
        if trams_added % 5 == 0:
            print(f"  Afegits {trams_added}/{metrics_ok.height} trams...")
    
    print(f"\nTotal trams afegits: {trams_added}/{metrics_ok.height}")
    
    # Afegim la llegenda
    legend_html = f'''
    <div style="position: fixed; 
                bottom: 50px; right: 50px; width: 220px; height: 180px; 
                background-color: white; border:2px solid #ccc; z-index:9999; 
                font-size:14px; padding: 10px; border-radius: 5px;
                box-shadow: 0 0 15px rgba(0,0,0,0.3);">
        <h4 style="margin-top: 0;">Llegenda MAE</h4>
        <p style="margin: 5px 0;">
            <span style="background-color: #2ecc71; padding: 5px 10px; border-radius: 3px; color: white;">
                <b>Verd</b>
            </span> 
            <br><small>MAE &lt; {mae_mean:.4f}</small>
        </p>
        <p style="margin: 5px 0;">
            <span style="background-color: #f39c12; padding: 5px 10px; border-radius: 3px; color: white;">
                <b>Taronja</b>
            </span>
            <br><small>{mae_mean:.4f} ≤ MAE &lt; {mae_mean*1.5:.4f}</small>
        </p>
        <p style="margin: 5px 0;">
            <span style="background-color: #e74c3c; padding: 5px 10px; border-radius: 3px; color: white;">
                <b>Vermell</b>
            </span>
            <br><small>MAE ≥ {mae_mean*1.5:.4f}</small>
        </p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Afegir control de capes
    folium.LayerControl().add_to(m)
    
    # Guardem el mapa
    Path(OUTPUT_MAP).parent.mkdir(parents=True, exist_ok=True)
    m.save(OUTPUT_MAP)
    
    print(f"\nEstadístiques:")
    print(f"  Trams verds (MAE baix):    {len([1 for row in metrics_ok.iter_rows(named=True) if row['mae'] < mae_mean])}")
    print(f"  Trams taronja (MAE mitjà): {len([1 for row in metrics_ok.iter_rows(named=True) if mae_mean <= row['mae'] < mae_mean*1.5])}")
    print(f"  Trams vermells (MAE alt):  {len([1 for row in metrics_ok.iter_rows(named=True) if row['mae'] >= mae_mean*1.5])}")
    
    return m


def main():
    """
    Funció principal.
    """
    create_mae_map()
    print(f"\nObre el mapa amb:")
    print(f"  open {OUTPUT_MAP}")


if __name__ == "__main__":
    main()
