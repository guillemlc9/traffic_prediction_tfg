"""
visualize_predictions.py
------------------------
Visualització de prediccions LSTM vs valors reals per un tram específic.

Aquest script genera un gràfic qualitatiu que mostra:
- Valors reals vs prediccions LSTM
- Durant 1-2 dies
- Per un tram específic (per defecte: tram 270)

NOTA: Aquest script carrega el model entrenat i genera prediccions noves
sobre una seqüència temporal contínua del tram especificat.
"""

# Fix per TensorFlow mutex issues en macOS
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import sys
from pathlib import Path
import json
import numpy as np
import polars as pl
import matplotlib
matplotlib.use('Agg')  # Backend sense GUI per evitar problemes
import matplotlib.pyplot as plt
from datetime import datetime

# Afegir el directori arrel al PYTHONPATH
root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root_dir))


def denormalize(normalized_values: np.ndarray, mean: float, std: float) -> np.ndarray:
    """Desnormalitza valors aplicant la transformació inversa de z-score."""
    return (normalized_values * std) + mean


def normalize(values: np.ndarray, mean: float, std: float) -> np.ndarray:
    """Normalitza valors aplicant z-score."""
    return (values - mean) / std


def create_predictions_for_tram(
    model,
    df_tram: pl.DataFrame,
    window_size: int,
    mean: float,
    std: float,
    n_days: int = 2
) -> tuple:
    """
    Crea prediccions per un tram específic durant N dies.
    
    Args:
        model: Model LSTM carregat
        df_tram: DataFrame amb les dades del tram (ordenades per timestamp)
        window_size: Mida de la finestra (36)
        mean: Mitjana per normalització
        std: Desviació estàndard per normalització
        n_days: Nombre de dies a visualitzar
    
    Returns:
        Tuple amb (timestamps, y_true, y_pred) en escala original
    """
    # Limitar a n_days
    n_samples_total = min(len(df_tram), n_days * 24 * 12)  # 12 mostres/hora
    
    # Necessitem window_size punts addicionals per començar
    n_needed = n_samples_total + window_size
    
    if len(df_tram) < n_needed:
        print(f"⚠️  Advertència: només hi ha {len(df_tram)} punts, es visualitzaran menys dies")
        n_samples_total = max(0, len(df_tram) - window_size)
    
    # Extreure valors
    values = df_tram["estatActual"].to_numpy()[:n_needed]
    timestamps = df_tram["timestamp"].to_list()[window_size:n_needed]
    
    # Normalitzar
    values_norm = normalize(values, mean, std)
    
    # Crear prediccions amb sliding window
    y_true_list = []
    y_pred_list = []
    
    for i in range(n_samples_total):
        # Finestra d'entrada
        window = values_norm[i:i+window_size].reshape(1, window_size, 1).astype(np.float32)
        
        # Valor real (següent timestep)
        y_true = values[i + window_size]
        
        # Predicció
        y_pred_norm = model.predict(window, verbose=0)[0, 0]
        y_pred = denormalize(np.array([y_pred_norm]), mean, std)[0]
        
        y_true_list.append(y_true)
        y_pred_list.append(y_pred)
    
    return timestamps, np.array(y_true_list), np.array(y_pred_list)


def plot_predictions(
    timestamps: list,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    tram_id: int,
    output_path: Path
):
    """
    Crea el gràfic de prediccions vs valors reals.
    
    Args:
        timestamps: Llista de timestamps
        y_true: Valors reals
        y_pred: Prediccions
        tram_id: ID del tram
        output_path: Ruta on guardar el gràfic
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Convertir timestamps a datetime si són strings
    if isinstance(timestamps[0], str):
        timestamps = [datetime.fromisoformat(ts.replace('Z', '+00:00')) for ts in timestamps]
    
    # Plot
    ax.plot(timestamps, y_true, label='Real', color='#2E86AB', linewidth=1.5, alpha=0.8)
    ax.plot(timestamps, y_pred, label='Predicció LSTM', color='#A23B72', linewidth=1.5, alpha=0.8, linestyle='--')
    
    # Configuració
    ax.set_xlabel('Temps', fontsize=12, fontweight='bold')
    ax.set_ylabel('Estat de Trànsit (1-5)', fontsize=12, fontweight='bold')
    ax.set_title(f'Prediccions LSTM vs Valors Reals - Tram {tram_id}', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Límits eix Y
    ax.set_ylim(0.5, 5.5)
    ax.set_yticks([1, 2, 3, 4, 5])
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Llegenda
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
    
    # Format dates
    fig.autofmt_xdate()
    
    # Ajustar layout
    plt.tight_layout()
    
    # Guardar
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Gràfic guardat a: {output_path}")
    
    plt.close()


def main():
    """
    Script principal per visualitzar prediccions.
    """
    import tensorflow as tf
    from src.data_prep.prepare_time_splits import get_temporal_splits
    
    # Configuració
    MODEL_DIR = "models/lstm/20251227_143405_lr1e-03_bs1024_u64_w36_h1"
    DATA_PATH = "data/processed/dataset_imputed_clean.parquet"
    NORM_PARAMS_PATH = "models/lstm/normalization_params.json"
    TARGET_TRAM = 270
    N_DAYS = 2
    WINDOW_SIZE = 36
    
    model_dir = Path(MODEL_DIR)
    
    print("=" * 60)
    print("VISUALITZACIÓ PREDICCIONS LSTM")
    print("=" * 60)
    print(f"Tram: {TARGET_TRAM}")
    print(f"Dies a visualitzar: {N_DAYS}\n")
    
    # 1. Carregar paràmetres de normalització
    print("1. Carregant paràmetres de normalització...")
    with open(NORM_PARAMS_PATH, 'r') as f:
        params = json.load(f)
    mean = params["mean"]
    std = params["std"]
    print(f"   Mean: {mean:.4f}, Std: {std:.4f}\n")
    
    # 2. Carregar model
    print("2. Carregant model LSTM...")
    model_path = model_dir / "best.keras"
    model = tf.keras.models.load_model(model_path)
    print(f"   ✓ Model carregat de {model_path}\n")
    
    # 3. Carregar dades del tram
    print(f"3. Carregant dades del tram {TARGET_TRAM}...")
    
    # Carregar dataset complet
    df = pl.read_parquet(DATA_PATH)
    
    # Obtenir splits temporals
    splits_info = get_temporal_splits(df)
    
    # Filtrar només test split i tram específic
    test_start = splits_info["val_end"]  # Test comença després de validació
    
    df_tram = df.filter(
        (pl.col("timestamp") >= test_start) &
        (pl.col("idTram") == TARGET_TRAM)
    ).sort("timestamp")
    
    print(f"   ✓ {len(df_tram)} registres trobats per aquest tram en test\n")
    
    # 4. Generar prediccions
    print(f"4. Generant prediccions per {N_DAYS} dies...")
    timestamps, y_true, y_pred = create_predictions_for_tram(
        model=model,
        df_tram=df_tram,
        window_size=WINDOW_SIZE,
        mean=mean,
        std=std,
        n_days=N_DAYS
    )
    print(f"   ✓ {len(timestamps)} prediccions generades\n")
    
    # 5. Crear visualització
    print("5. Creant visualització...")
    output_path = model_dir / "visualizations" / f"predictions_tram_{TARGET_TRAM}.png"
    plot_predictions(timestamps, y_true, y_pred, TARGET_TRAM, output_path)
    
    # 6. Estadístiques
    print("\n" + "=" * 60)
    print("ESTADÍSTIQUES DE LA VISUALITZACIÓ")
    print("=" * 60)
    mae = np.mean(np.abs(y_pred - y_true))
    rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
    print(f"MAE (mostres visualitzades):  {mae:.4f}")
    print(f"RMSE (mostres visualitzades): {rmse:.4f}")
    print(f"Rang valors reals:  [{y_true.min():.2f}, {y_true.max():.2f}]")
    print(f"Rang prediccions:   [{y_pred.min():.2f}, {y_pred.max():.2f}]")
    print("=" * 60)
    
    print("\n✅ Visualització completada!")
    print(f"\nEl gràfic mostra com el model LSTM captura:")
    print("  - Transicions suaus entre estats de trànsit")
    print("  - Canvis sobtats en les condicions de trànsit")
    print(f"  - Patrons temporals del tram {TARGET_TRAM}")


if __name__ == "__main__":
    main()
