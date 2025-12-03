# src

Aquest directori conté el codi Python reutilitzable del projecte de predicció de trànsit. S’organitza en mòduls per preparar les dades i entrenar models.

## Estructura

- **arima/** – scripts relacionats amb el model ARIMA de base:
  - `auto_arima_params.py`: cerca automàticament els millors paràmetres (p,d,q) per a cada sèrie.
  - `train_arima_baseline.py`: entrena models ARIMA(1,1,1) per a cada tram seleccionat i desa els models i mètriques.
  - `evaluate_arima_baseline.py`: avalua els models entrenats i genera informes i figures dins de `reports/arima/`.
  - `visualize_mae_map.py`: crea mapes i gràfics de l’error absolut mitjà (MAE) per tram.
  - `seleccio_trams.py`: utilitats per seleccionar els trams d’interès.

- **data_prep/** – eines de preparació de dades:
  - `create_parquet.py`: combina els fitxers CSV de trànsit i genera un Parquet amb els timestamps normalitzats.
  - `gaps.py`: identifica buits temporals en les sèries i els classifica per durada.
  - `imputation.py`: aplica tècniques d’imputació (curta/mitjana/llarga) per omplir els buits detectats.
  - `prepare_time_splits.py`: defineix els splits temporals consistents (train/val/test) i selecciona els trams que s’analitzaran.
  - `centroid.py`: utilitats per calcular la posició geogràfica dels trams.
  - `__init__.py`: permet que aquest directori es pugui importar com a paquet.

## Ús

Importa les funcions directament, per exemple:

```python
from src.data_prep.prepare_time_splits import get_temporal_splits, prepare_tram_series
