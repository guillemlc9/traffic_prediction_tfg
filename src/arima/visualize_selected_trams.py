"""
visualize_selected_trams.py
----------------------------
Crea un mapa simple de Barcelona mostrant només els 30 trams seleccionats.
"""

import sys
from pathlib import Path

# Afegir el directori arrel al PYTHONPATH
root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root_dir))

import polars as pl
import folium
from src.data_prep.prepare_time_splits import SELECTED_TRAMS
import warnings
warnings.filterwarnings('ignore')


# Paths
TRAMS_PATH = "data/transit_relacio_trams_format_long.csv"
OUTPUT_MAP = "reports/arima/selected_trams_map.html"


def create_selected_trams_map():
    """
    Crea un mapa simple amb els 30 trams seleccionats.
    """
    print("=" * 60)
    print("GENERANT MAPA DELS 30 TRAMS SELECCIONATS")
    print("=" * 60)
    
    # Carregar dades de trams
    print("\nCarregant dades de trams...")
    trams_df = pl.read_csv(TRAMS_PATH)
    
    # Crear mapa centrat a Barcelona
    barcelona_center = [41.3851, 2.1734]
    m = folium.Map(
        location=barcelona_center,
        zoom_start=13,
        tiles='CartoDB positron'
    )
    
    # Afegir trams al mapa
    print(f"\nAfegint {len(SELECTED_TRAMS)} trams al mapa...")
    trams_added = 0
    
    for tram_id in SELECTED_TRAMS:
        # Obtenir coordenades del tram
        tram_coords = trams_df.filter(pl.col('Tram') == tram_id)
        
        if tram_coords.height == 0:
            print(f"  ⚠️  Tram {tram_id}: No s'han trobat coordenades")
            continue
        
        # Convertir a pandas per facilitar l'ús amb folium
        tram_coords_pd = tram_coords.to_pandas()
        
        # Crear llista de coordenades (lat, lon)
        coordinates = list(zip(
            tram_coords_pd['Latitud'],
            tram_coords_pd['Longitud']
        ))
        
        # Obtenir descripció del tram
        descripcio = tram_coords_pd['Descripció'].iloc[0] if len(tram_coords_pd) > 0 else f"Tram {tram_id}"
        
        # Crear popup amb informació bàsica
        popup_html = f"""
        <div style="font-family: Arial; width: 200px;">
            <h4 style="margin: 0; color: #3498db;">Tram {tram_id}</h4>
            <p style="margin: 5px 0;"><b>{descripcio}</b></p>
        </div>
        """
        
        # Afegir línia al mapa (color blau uniforme)
        folium.PolyLine(
            coordinates,
            color='#3498db',  # Blau
            weight=4,
            opacity=0.7,
            popup=folium.Popup(popup_html, max_width=250),
            tooltip=f"Tram {tram_id}"
        ).add_to(m)
        
        trams_added += 1
        
        if trams_added % 5 == 0:
            print(f"  Afegits {trams_added}/{len(SELECTED_TRAMS)} trams...")
    
    print(f"\n✓ Total trams afegits: {trams_added}/{len(SELECTED_TRAMS)}")
    
    # Afegir títol al mapa
    title_html = '''
    <div style="position: fixed; 
                top: 10px; left: 50px; width: 300px; height: 60px; 
                background-color: white; border:2px solid #3498db; z-index:9999; 
                font-size:16px; padding: 10px; border-radius: 5px;
                box-shadow: 0 0 15px rgba(0,0,0,0.3);">
        <h3 style="margin: 0; color: #3498db;">30 Trams Seleccionats</h3>
        <p style="margin: 5px 0; font-size: 12px;">Barcelona - Xarxa de Trànsit</p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(title_html))
    
    # Afegir control de capes
    folium.LayerControl().add_to(m)
    
    # Guardar mapa
    Path(OUTPUT_MAP).parent.mkdir(parents=True, exist_ok=True)
    m.save(OUTPUT_MAP)
    
    print(f"\n✅ Mapa guardat: {OUTPUT_MAP}")
    
    return m


def main():
    """
    Funció principal.
    """
    create_selected_trams_map()
    print("\n✅ Procés completat!")
    print(f"\nObre el mapa amb:")
    print(f"  open {OUTPUT_MAP}")


if __name__ == "__main__":
    main()
